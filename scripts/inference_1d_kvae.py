import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from audiotools import AudioSignal
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AudioDataset, save_audio
from kvae.models import KVAEAudio
from metrics.audio_metrics import (
    MelSpectrogramDistance,
    MultiScaleMFCCDistance,
    MultiScaleSTFTDistance,
    SISDRDistance,
    WaveformL1Distance,
)
from metrics.ccpc import compute_ccpc
from scripts.common_utils import set_seed_and_optimal_cuda_env

MODEL_PATHS = {
    "KVAE-Audio": "kandinskylab/KVAE-Audio",
    "KVAE-Audio-Stereo": "kandinskylab/KVAE-Audio-Stereo",
}


def match_model_channels(audio_data: torch.Tensor, model_channels: int) -> torch.Tensor:
    """Make [B, C, T] match the model width.

    A mono file into a wider model is duplicated. Extra channels are dropped.
    """
    channels = audio_data.shape[1]
    if channels == model_channels:
        return audio_data
    if channels == 1 and model_channels > 1:
        return audio_data.repeat(1, model_channels, 1)
    if channels > model_channels:
        return audio_data[:, :model_channels]
    pad = audio_data[:, -1:].repeat(1, model_channels - channels, 1)
    return torch.cat([audio_data, pad], dim=1)


@torch.no_grad()
def run_audio_inference(
    vae: nn.Module,
    device: torch.device,
    data_dir: str,
    batch_size: int = 1,
    saving_folder: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    sample_posterior: bool = False,
) -> dict[str, float]:
    """Run KVAEAudio reconstruction, metrics, and optional WAV saving.

    Audio files are loaded without resampling or normalization. Because files
    commonly have different lengths, batch size 1 is the safe default.
    """
    dataset = AudioDataset(root_path=data_dir)
    if len(dataset) == 0:
        raise ValueError(f"No audio found in {data_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    distances = {
        "waveform_l1": WaveformL1Distance().to(device),
        "stft": MultiScaleSTFTDistance().to(device),
        "mel": MelSpectrogramDistance().to(device),
        "mfcc": MultiScaleMFCCDistance().to(device),
        "sisdr": SISDRDistance().to(device),
    }
    values = {name: [] for name in distances}
    values["ccpc"] = []

    print(f"Starting inference on {len(dataset)} audio files...")

    for data in tqdm(dataloader, desc="Processing"):
        audio = data["audio"].to(dtype=dtype, device=device)
        sample_rates = data["sample_rate"].tolist()
        if len(set(sample_rates)) != 1:
            raise ValueError("All audio files in a batch must use one sample rate")
        sample_rate = int(sample_rates[0])
        batch, num_input_channels, length = audio.shape
        model_channels = int(getattr(vae, "num_channels", num_input_channels))
        # Mono model + stereo: encode each channel, glue on write.
        batched_channels = model_channels == 1 and num_input_channels > 1
        # Stereo model + mono: duplicate to L/R, average on write.
        average_channels = model_channels > 1 and num_input_channels == 1
        if batched_channels:
            model_audio = audio.reshape(batch * num_input_channels, 1, length)
        else:
            model_audio = match_model_channels(audio, model_channels)

        posterior = vae.encode(model_audio).latent_dist
        latent = posterior.sample() if sample_posterior else posterior.mode()
        reconstructions = vae.decode(latent)
        if batched_channels:
            reconstructions = reconstructions.reshape(batch, num_input_channels, -1)
        elif average_channels:
            reconstructions = reconstructions.mean(dim=1, keepdim=True)
        reconstructions = reconstructions[..., :length]

        for path, reference, reconstruction, real_len in zip(
            data["paths"], audio, reconstructions, data["real_len"]
        ):
            valid_length = min(
                int(real_len), reference.shape[-1], reconstruction.shape[-1]
            )
            reference = reference[..., :valid_length].unsqueeze(0).float()
            reconstruction = reconstruction[..., :valid_length].unsqueeze(0).float()
            channels = min(reference.shape[1], reconstruction.shape[1])
            reference = reference[:, :channels]
            reconstruction = reconstruction[:, :channels]

            reference_signal = AudioSignal(reference, sample_rate)
            reconstruction_signal = AudioSignal(reconstruction, sample_rate)

            values["waveform_l1"].append(
                distances["waveform_l1"](reference_signal, reconstruction_signal).detach().cpu()
            )
            values["stft"].append(
                distances["stft"](reference_signal, reconstruction_signal).detach().cpu()
            )
            values["mel"].append(
                distances["mel"](reference_signal, reconstruction_signal).detach().cpu()
            )
            values["mfcc"].append(
                distances["mfcc"](reference_signal, reconstruction_signal).detach().cpu()
            )
            values["sisdr"].append(
                -distances["sisdr"](reference_signal, reconstruction_signal).detach().cpu()
            )
            values["ccpc"].append(
                compute_ccpc(reference, reconstruction).mean().detach().cpu()
            )

            if saving_folder is not None:
                save_audio(
                    tensor=reconstruction,
                    reference_tensor=reference,
                    sample_rate=sample_rate,
                    save_dir_path=saving_folder,
                    filename=Path(path).name,
                )

    results = {
        name: torch.stack(metric_values).mean().item()
        for name, metric_values in values.items()
    }

    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"Waveform L1: {results['waveform_l1']:.8f}")
    print(f"STFT dist  : {results['stft']:.8f}")
    print(f"Mel dist   : {results['mel']:.8f}")
    print(f"MFCC dist  : {results['mfcc']:.8f}")
    print(f"SI-SDR (dB): {results['sisdr']:.4f}")
    print(f"CCPC       : {results['ccpc']:.4f}")
    print("=" * 40)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KVAE 1D inference")
    parser.add_argument("--device", type=int, default=0, help="GPU number")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="./assets/audio_test",
        help="Path to an audio file or folder",
    )
    parser.add_argument("--bs", type=int, default=1, help="DataLoader batch size")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PATHS),
        default="KVAE-Audio-Stereo",
        help="KVAE-Audio has 64 latent channels. KVAE-Audio-Stereo has 128 latent channels.",
    )
    parser.add_argument(
        "--saving_folder",
        type=str,
        default="./outputs/audio",
        help="Folder for saved reconstructions",
    )
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        help="Sample the posterior instead of using its mode",
    )
    cli_args = parser.parse_args()

    set_seed_and_optimal_cuda_env(111)

    device = torch.device(f"cuda:{cli_args.device}")
    dtype = torch.float32
    vae = (
        KVAEAudio.from_pretrained(MODEL_PATHS[cli_args.model])
        .eval()
        .to(device=device, dtype=dtype)
    )

    run_audio_inference(
        vae=vae,
        device=device,
        data_dir=cli_args.dataset_folder,
        batch_size=cli_args.bs,
        saving_folder=cli_args.saving_folder,
        dtype=dtype,
        sample_posterior=cli_args.sample_posterior,
    )
