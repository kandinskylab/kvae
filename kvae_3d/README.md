# K-VAE 3D

## Setup

Install requirements:
```sh 
pip install -r requirements.txt
```

## Test

For simple test, run 
```sh
python inference.py --frames 999
```

To use optimized compiled encoder version, run (max duration 257 frames):
```sh
python inference.py --frames 257 --optim
```
