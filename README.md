<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/kvae-white.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/kvae-black.png">
  </picture>
</div>
<div align="center">
  <a href="https://habr.com/ru/companies/sberbank/articles/966450/">Habr</a> | <a href="https://kandinskylab.ai/">Project Page</a> | Technical Report (soon) | 🤗 <a href=https://huggingface.co/kandinskylab/KVAE-3D-1.0> KVAE-3D </a> / <a href=https://huggingface.co/kandinskylab/KVAE-2D-1.0> KVAE-2D </a> 
</div>

In this repository, we provide tokenizers for image and video diffusion models: 
KVAE 2D and KVAE 3D respectively.

## Setup

Install requirements:
```sh 
pip install -r requirements.txt
```

## KVAE 2D inference
Minimum example for 2d model inference in presented in  


## KVAE 3D inference
For simple test, run
```sh
python inference.py --frames 999
```

To use optimized compiled encoder version, run (max duration 257 frames):
```sh
python inference.py --frames 257 --optim
```
