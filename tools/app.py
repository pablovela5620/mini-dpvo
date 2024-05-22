import gradio as gr

# import spaces
from gradio_rerun import Rerun
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
import uuid
import mmcv

from mini_dpvo.api.inference import run
from mini_dpvo.config import cfg as base_cfg

base_cfg.merge_from_file("config/fast.yaml")
base_cfg.BUFFER_SIZE = 2048


def create_blueprint(image_name_list: list[str], log_path: Path) -> rrb.Blueprint:
    # dont show 2d views if there are more than 4 images as to not clutter the view
    if len(image_name_list) > 4:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin=f"{log_path}"),
            ),
            collapse_panels=True,
        )
    else:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                contents=[
                    rrb.Spatial3DView(origin=f"{log_path}"),
                    rrb.Vertical(
                        contents=[
                            rrb.Spatial2DView(
                                origin=f"{log_path}/camera_{i}/pinhole/",
                                contents=[
                                    "+ $origin/**",
                                ],
                            )
                            for i in range(len(image_name_list))
                        ]
                    ),
                ],
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    return blueprint


# @spaces.GPU
def predict(video_file_path: str, stride: int) -> tuple[str, str]:
    # check if is list or string and if not raise error
    if not isinstance(video_file_path, str):
        raise gr.Error(
            f"Something is wrong with your input video, got: {type(video_file_path)}"
        )

    uuid_str = str(uuid.uuid4())
    filename = Path(f"/tmp/gradio/{uuid_str}.rrd")
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    rr.init(f"{uuid_str}")

    dpvo_pred, time_taken = run(
        cfg=base_cfg,
        network_path="checkpoints/dpvo.pth",
        imagedir=video_file_path,
        calib="data/calib/iphone.txt",
        stride=stride,
        skip=0,
        vis_during=True,
    )

    # blueprint: rrb.Blueprint = create_blueprint(image_name_list, log_path)
    # rr.send_blueprint(blueprint)

    rr.set_time_sequence("sequence", 0)
    # log_optimized_result(optimized_results, log_path)
    rr.save(filename.as_posix())
    return filename.as_posix(), f"Total time: {time_taken:.2f}s"


def on_file_upload(video_file_path: str) -> None:
    video_reader = mmcv.VideoReader(video_file_path)
    video_info = f"""
    **Video Info:**
    - Number of Frames: {video_reader.frame_cnt}
    - FPS: {round(video_reader.fps)}
    """
    return video_info


with gr.Blocks(
    css=""".gradio-container {margin: 0 !important; min-width: 100%};""",
    title="Mini-DPVO Demo",
) as demo:
    # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
    gr.HTML('<h2 style="text-align: center;">Mini-DPVO Demo</h2>')
    gr.HTML(
        '<p style="text-align: center;">Unofficial DPVO demo using the mini-dpvo pip package</p>'
    )
    gr.HTML(
        '<p style="text-align: center;">Learn more about mini-dpvo here <a href="https://github.com/pablovela5620/mini-dpvo">here</a></p>'
    )
    with gr.Tab(label="Video Input"):
        with gr.Column():
            with gr.Row():
                video_input = gr.File(
                    height=300,
                    file_count="single",
                    file_types=[".mp4", ".mov"],
                    label="Video",
                )
                with gr.Column():
                    video_info = gr.Markdown(
                        value="""
                    **Video Info:**
                    """
                    )
                    time_taken = gr.Textbox(label="Time Taken")
            with gr.Accordion(label="Advanced", open=False):
                stride = gr.Slider(
                    label="Stride",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=2,
                )
            run_btn_single = gr.Button("Run")
            rerun_viewer_single = Rerun(height=900)
            run_btn_single.click(
                fn=predict,
                inputs=[video_input, stride],
                outputs=[rerun_viewer_single, time_taken],
            )

            video_input.upload(
                fn=on_file_upload, inputs=[video_input], outputs=[video_info]
            )


demo.launch(share=False)
