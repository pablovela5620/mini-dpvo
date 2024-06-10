# Mini-DPVO
A miniature version of [DPVO](https://github.com/princeton-vl/DPVO). Using Rerun viewer, Pixi and Gradio for easy use
<p align="center">
  <img src="media/mini-dpvo.gif" alt="example output" width="720" />
</p>


## Installation
Easily installable via [Pixi](https://pixi.sh/latest/).
```bash
git clone https://github.com/pablovela5620/mini-dpvo.git
cd mini-dpvo
pixi run post-install
```

## Demo
<a target="_blank" href="https://lightning.ai/pablovelagomez1/studios/mini-dpvo">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>


To run the gradio frontend
```
pixi run app
```

To run using just the rerun visualizer
```
pixi run demo
```

look in the `pixi.toml` file to see exactly what each command does under `tasks`


## Acknowledgements
Original Code and paper form [DPVO](https://github.com/princeton-vl/DPVO)
```
@article{teed2023deep,
  title={Deep Patch Visual Odometry},
  author={Teed, Zachary and Lipson, Lahav and Deng, Jia},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
