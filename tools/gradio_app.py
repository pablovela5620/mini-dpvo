from mini_dpvo.gradio_ui.dpvo_ui import dpvo_block
import gradio as gr

with gr.Blocks() as demo:
    dpvo_block.render()

demo.launch()
