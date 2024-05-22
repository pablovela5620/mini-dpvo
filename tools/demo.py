from argparse import ArgumentParser
import rerun as rr
from mini_dpvo.api.inference import run
from mini_dpvo.config import cfg as base_cfg


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--network-path", type=str, default="checkpoints/dpvo.pth")
    parser.add_argument("--imagedir", type=str)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--buffer", type=int, default=2048)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--vis-during", action="store_true")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini_dpvo")

    base_cfg.merge_from_file(args.config)
    base_cfg.BUFFER_SIZE = args.buffer

    print("Running with config...")
    print(base_cfg)

    run(
        base_cfg,
        args.network_path,
        args.imagedir,
        args.calib,
        args.stride,
        args.skip,
        vis_during=args.vis_during,
    )
    rr.script_teardown(args)
