# Inference

This guide covers command-line scripts and the direct Python API for audio, image and video reconstruction. Follow the [README quick start](../../README.md#quick-start) and run commands from the repository root.

## Common model API

The public classes are:

```python
from kvae import KVAEAudio, KVAEImage, KVAEVideo
```

All three support Hugging Face Hub loading through `from_pretrained` and expose a diffusers-style encoder result:

```python
posterior = model.encode(input_tensor, ...).latent_dist
latent = posterior.mode()       # deterministic reconstruction
latent = posterior.sample()     # stochastic latent sample
reconstruction = model.decode(latent, ...)
```

Use `model.eval()` and `torch.no_grad()` for inference. The released image and video scripts use `bfloat16`; audio uses `float32` to preserve parity with the original implementation.

## Command-line scripts

| Modality | Module | Main metrics |
| --- | --- | --- |
| Audio | `scripts.inference_1d_kvae` | waveform L1, multi-scale STFT, mel distance, SI-SDR |
| Image | `scripts.inference_2d_kvae` | PSNR, LPIPS |
| Video | `scripts.inference_3d_kvae` | PSNR, SSIM, LPIPS |

Use `python -m ... --help` to inspect every current option.

### Audio CLI

```bash
python -m scripts.inference_1d_kvae \
    --device 0 \
    --dataset_folder ./assets/audio_test \
    --model KVAE-Audio \
    --bs 1 \
    --saving_folder ./outputs/audio
```

`--dataset_folder` may point to one audio file or a directory, which is searched recursively. Use `--sample_posterior` to sample the latent distribution; without it, inference uses the deterministic posterior mode.

The released checkpoint expects mono audio at 48 kHz. `read_audio` and `AudioDataset` intentionally preserve the source sample rate, channel count, length, and amplitude. They do not resample, downmix, crop, pad, or normalizeinput files. Variable-length files require `--bs 1` unless a custom collate
function is introduced.

Metrics are calculated on the raw decoder reconstruction. Before a WAV is written, `save_audio` matches its loudness to the input signal on the current device and only then transfers it to CPU.

### Image CLI

```bash
python -m scripts.inference_2d_kvae \
    --device 0 \
    --dataset_folder ./assets/image_test \
    --model KVAE_1.0 \
    --saving_folder ./outputs/images
```

The current dataset script searches for PNG images. Add `--img_size H,W` to resize images to a common spatial size and allow batching. Without a requested size, batch size is forced to one.

Image inference runs in `bfloat16`. Inputs are normalized to approximately `[-1, 1]`, reconstructions can be written as PNG files, and the script reports PSNR and LPIPS.

### Video CLI

```bash
python -m scripts.inference_3d_kvae \
    --device 0 \
    --dataset_folder ./assets/video_test \
    --model KVAE_2.0-t4s8 \
    --input_norm m11 \
    --seg_len 16 \
    --saving_folder ./outputs/video
```

Available model names are `KVAE_1.0`, `KVAE_2.0-t4s8`, and `KVAE_2.0-t4s16`. The current inference script is configured for directories of PNG frames. Each video is represented by one subdirectory. `seg_len` controls temporal processing and accepts 4, 8, or 16.

Temporal context is cached inside the causal convolution, residual, and sampling blocks while a video is split into chunks. The cache is mutable model state: do not run independent videos concurrently or interleave chunks on the same `KVAEVideo` instance. Use one model instance per concurrent stream. Sequential calls are safe because `encode` and `decode` reset all block caches before and after processing.

The script reports frame-wise PSNR, SSIM, and LPIPS aggregated per video. See [Video metrics](#video-metrics) for the exact input layout, normalization, and averaging behavior.

## Direct Python API

### Audio reconstruction

```python
from pathlib import Path

import torch

from data import read_audio, save_audio
from kvae import KVAEAudio

device = torch.device("cuda:0")
model = (
    KVAEAudio.from_pretrained("kandinskylab/KVAE-Audio")
    .eval()
    .to(device=device, dtype=torch.float32)
)

input_path = Path("assets/audio_test/98537484.wav")
signal = read_audio(input_path).to(device)
audio = signal.audio_data.float()

with torch.no_grad():
    posterior = model.encode(audio, signal.sample_rate).latent_dist
    latent = posterior.mode()
    reconstruction = model.decode(latent)[..., : audio.shape[-1]]

save_audio(
    tensor=reconstruction,
    reference_tensor=audio,
    sample_rate=signal.sample_rate,
    save_dir_path="outputs/audio",
    filename=input_path.name,
)
```

The explicit crop restores the original sample count after the model pads the input to its temporal hop length.

### Image reconstruction

```python
import torch

from data import read_image
from kvae import KVAEImage

device = torch.device("cuda:0")
model = (
    KVAEImage.from_pretrained("kandinskylab/KVAE-2D-1.0")
    .eval()
    .to(device=device, dtype=torch.bfloat16)
)
image = read_image("assets/image_test/0002.png").unsqueeze(0)
image = image.to(device=device, dtype=torch.bfloat16)

with torch.no_grad():
    latent = model.encode(image).latent_dist.mode()
    reconstruction = model.decode(latent).clip(-1, 1)
```

Image tensors use `[batch, channels, height, width]` layout and project loaders normalize pixels to approximately `[-1, 1]`.

### Video reconstruction

```python
import torch

from data import VideoReader
from kvae import KVAEVideo

device = torch.device("cuda:0")
model = (
    KVAEVideo.from_pretrained("kandinskylab/KVAE-3D-2.0-t4s8")
    .eval()
    .to(device=device, dtype=torch.bfloat16)
)
reader = VideoReader(stream_pattern="*.png", input_norm="m11")
video = reader.read_video("assets/video_test/31")["frames"].unsqueeze(0)
video = video.to(device=device, dtype=torch.bfloat16)

with torch.no_grad():
    latent = model.encode(video, seg_len=16).latent_dist.mode()
    reconstruction = model.decode(latent, seg_len=16).clip(-1, 1)
```

Input video tensors use `[batch, channels, time, height, width]` layout. Spatial and temporal truncation is performed by the current data utilities to match model compression constraints.

## Audio data and metrics

Public audio utilities:

```python
from data import AudioDataset, read_audio, save_audio
from metrics import (
    MelSpectrogramDistance,
    MultiScaleSTFTDistance,
    SISDRDistance,
    WaveformL1Distance,
)
```

The metric classes are stateless PyTorch modules. `SISDRDistance` returns negative SI-SDR so that lower values behave like a distance; the inference script negates it for display as conventional higher-is-better SI-SDR in decibels.

## Video metrics

Public video metrics are TorchMetrics-compatible classes:

```python
from metrics import VideoLPIPS, VideoPSNR, VideoSSIM
from torchmetrics import MetricCollection

metrics = MetricCollection(
    {
        "psnr": VideoPSNR(data_range=(-1, 1), metric_chank_size=10),
        "ssim": VideoSSIM(data_range=(-1, 1), metric_chank_size=10),
        "lpips": VideoLPIPS(net_type="alex", metric_chank_size=10),
    }
).to(device)

# One video per update; tensors have [T, C, H, W] layout.
metrics.update(reconstruction.float(), reference.float())
results = metrics.compute()

print(results["psnr_dataset_mean"])
print(results["ssim_dataset_mean"])
print(results["lpips_dataset_mean"])
```

PSNR and SSIM are higher-is-better; LPIPS is lower-is-better. Each image metric is evaluated frame by frame. Frames are averaged into one value per video, then those video values are averaged with equal weight into `dataset_mean`. The corresponding per-video values remain available as `psnr_metric_per_video`, `ssim_metric_per_video`, and `lpips_metric_per_video`.

`metric_chank_size` is the current public parameter name. It limits how many frames are passed to the underlying image metric at once and therefore controls metric memory use; it is independent of the model's `seg_len`.

To reproduce the CLI results exactly, apply the same preparation used by `scripts.inference_3d_kvae`: `quant_renormalization` performs an 8-bit quantization round-trip, converts the selected `input_norm` to `[-1, 1]`, and changes batched tensors from `[B, C, T, H, W]` to `[B, T, C, H, W]`. Its transpose is in-place, so pass a clone if the original layout is still needed. Update the metrics once for every video after cropping both tensors to the same valid `real_len`. Passing raw decoder values directly can produce slightly different scores.