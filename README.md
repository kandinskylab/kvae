# K-VAE 1.0
<div align="center">
  <a href="https://habr.com/ru/companies/sberbank/articles/951800/">Habr</a> | <a href="https://kandinskylab.ai/">Project Page</a> | Technical Report (soon) | 🤗 <a href=https://huggingface.co/collections/kandinskylab/kandinsky-50-video-lite> Video Lite </a> / <a href=https://huggingface.co/collections/kandinskylab/kandinsky-50-video-pro> Video Pro </a> / <a href=https://huggingface.co/collections/kandinskylab/kandinsky-50-image-lite> Image Lite </a> | <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky5"> 🤗 Diffusers </a>  | <a href="https://github.com/kandinskylab/kandinsky-5/blob/main/comfyui/README.md">ComfyUI</a>
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