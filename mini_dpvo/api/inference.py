import numpy as np
import os
import torch
from pathlib import Path
from multiprocessing import Process, Queue
from yacs.config import CfgNode

from mini_dpvo.utils import Timer
from mini_dpvo.dpvo import DPVO
from mini_dpvo.stream import image_stream, video_stream

import rerun as rr
from jaxtyping import UInt8, Float64, Float32
from scipy.spatial.transform import Rotation
from dataclasses import dataclass

from timeit import default_timer as timer


@dataclass
class DPVOPrediction:
    final_poses: Float32[torch.Tensor, "num_keyframes 7"]  # noqa: F722
    tstamps: Float64[torch.Tensor, "num_keyframes"]  # noqa: F821
    final_points: Float32[torch.Tensor, "buffer_size*num_patches 3"]  # noqa: F722
    final_colors: UInt8[torch.Tensor, "buffer_size num_patches 3"]  # noqa: F722


def log_trajectory(
    parent_log_path: Path,
    poses: Float32[torch.Tensor, "buffer_size 7"],
    points: Float32[torch.Tensor, "buffer_size*num_patches 3"],
    colors: UInt8[torch.Tensor, "buffer_size num_patches 3"],
    intri_np: Float64[np.ndarray, "4"],
    bgr_hw3: UInt8[np.ndarray, "h w 3"],
):
    cam_log_path = f"{parent_log_path}/camera"
    rr.log(f"{cam_log_path}/pinhole/image", rr.Image(bgr_hw3[..., ::-1]))
    rr.log(
        f"{cam_log_path}/pinhole",
        rr.Pinhole(
            height=bgr_hw3.shape[0],
            width=bgr_hw3.shape[1],
            focal_length=[intri_np[0], intri_np[1]],
            principal_point=[intri_np[2], intri_np[3]],
        ),
    )

    poses_mask = ~(poses[:, :6] == 0).all(dim=1)
    points_mask = ~(points == 0).all(dim=1)

    nonzero_poses = poses[poses_mask]
    nonzero_points = points[points_mask]

    last_index = nonzero_poses.shape[0] - 1
    # get last non-zero pose, and the index of the last non-zero pose
    quat_pose = nonzero_poses[last_index].numpy(force=True)
    trans_quat = quat_pose[:3]
    rotation_quat = Rotation.from_quat(quat_pose[3:])

    mat3x3 = rotation_quat.as_matrix()
    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(translation=trans_quat, mat3x3=mat3x3, from_parent=True),
    )

    # outlier removal
    trajectory_center = np.median(nonzero_poses[:, :3].numpy(force=True), axis=0)
    radii = lambda a: np.linalg.norm(a - trajectory_center, axis=1)
    points_np = nonzero_points.view(-1, 3).numpy(force=True)
    colors_np = colors.view(-1, 3)[points_mask].numpy(force=True)
    inlier_mask = (
        radii(points_np) < radii(nonzero_poses[:, :3].numpy(force=True)).max() * 5
    )
    points_filtered = points_np[inlier_mask]
    colors_filtered = colors_np[inlier_mask]

    # log all points and colors at the same time
    rr.log(
        f"{parent_log_path}/pointcloud",
        rr.Points3D(
            positions=points_filtered,
            colors=colors_filtered,
        ),
    )


def log_final(
    parent_log_path: Path,
    final_poses: Float32[torch.Tensor, "num_keyframes 7"],
    tstamps: Float64[torch.Tensor, "num_keyframes"],  # noqa: F821
    final_points: Float32[torch.Tensor, "buffer_size*num_patches 3"],
    final_colors: UInt8[torch.Tensor, "buffer_size num_patches 3"],
):
    for idx, (pose_quat, tstamp) in enumerate(zip(final_poses, tstamps)):
        cam_log_path = f"{parent_log_path}/camera_{idx}"
        trans_quat = pose_quat[:3]
        R_33 = Rotation.from_quat(pose_quat[3:]).as_matrix()
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(translation=trans_quat, mat3x3=R_33, from_parent=False),
        )


def create_reader(
    imagedir: str, calib: str, stride: int, skip: int, queue: Queue
) -> Process:
    if os.path.isdir(imagedir):
        reader = Process(
            target=image_stream, args=(queue, imagedir, calib, stride, skip)
        )
    else:
        reader = Process(
            target=video_stream, args=(queue, imagedir, calib, stride, skip)
        )

    return reader


@torch.no_grad()
def run(
    cfg: CfgNode,
    network_path: str,
    imagedir: str,
    calib: str,
    stride: int = 1,
    skip: int = 0,
    vis_during: bool = True,
    timeit: bool = False,
) -> tuple[DPVOPrediction, float]:
    slam = None
    queue = Queue(maxsize=8)
    reader: Process = create_reader(imagedir, calib, stride, skip, queue)
    reader.start()

    if vis_during:
        parent_log_path = Path("world")
        rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)

    start = timer()

    while True:
        t: int
        bgr_hw3: UInt8[np.ndarray, "h w 3"]
        intri_np: Float64[np.ndarray, "4"]
        (t, bgr_hw3, intri_np) = queue.get()
        # queue will have a (-1, image, intrinsics) tuple when the reader is done
        if t < 0:
            break

        if vis_during:
            rr.set_time_sequence(timeline="timestep", sequence=t)

        bgr_3hw: UInt8[torch.Tensor, "h w 3"] = (
            torch.from_numpy(bgr_hw3).permute(2, 0, 1).cuda()
        )
        intri_torch: Float64[torch.Tensor, "4"] = torch.from_numpy(intri_np).cuda()

        if slam is None:
            slam = DPVO(cfg, network_path, ht=bgr_3hw.shape[1], wd=bgr_3hw.shape[2])

        with Timer("SLAM", enabled=timeit):
            slam(t, bgr_3hw, intri_torch)

        if slam.is_initialized and vis_during:
            poses: Float32[torch.Tensor, "buffer_size 7"] = slam.poses_
            points: Float32[torch.Tensor, "buffer_size*num_patches 3"] = slam.points_
            colors: UInt8[torch.Tensor, "buffer_size num_patches 3"] = slam.colors_
            log_trajectory(parent_log_path, poses, points, colors, intri_np, bgr_hw3)

    for _ in range(12):
        slam.update()

    total_time: float = timer() - start
    print(f"Total time: {total_time:.2f}s")

    reader.join()

    final_poses: Float32[torch.Tensor, "num_keyframes 7"]
    tstamps: Float64[torch.Tensor, "num_keyframes"]  # noqa: F821

    final_poses, tstamps = slam.terminate()
    final_points: Float32[torch.Tensor, "buffer_size*num_patches 3"] = slam.points_
    final_colors: UInt8[torch.Tensor, "buffer_size num_patches 3"] = slam.colors_
    dpvo_pred = DPVOPrediction(
        final_poses=final_poses,
        tstamps=tstamps,
        final_points=final_points,
        final_colors=final_colors,
    )
    return dpvo_pred, total_time
