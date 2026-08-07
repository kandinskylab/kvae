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
    MultiScaleSTFTDistance,
    SISDRDistance,
    WaveformL1Distance,
)
from scripts.common_utils import set_seed_and_optimal_cuda_env


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
        "sisdr": SISDRDistance().to(device),
    }
    values = {name: [] for name in distances}

    print(f"Starting inference on {len(dataset)} audio files...")

    for data in tqdm(dataloader, desc="Processing"):
        audio = data["audio"].to(dtype=dtype, device=device)
        sample_rates = data["sample_rate"].tolist()
        if len(set(sample_rates)) != 1:
            raise ValueError("All audio files in a batch must use one sample rate")
        sample_rate = int(sample_rates[0])

        posterior = vae.encode(audio, sample_rate).latent_dist
        latent = posterior.sample() if sample_posterior else posterior.mode()
        reconstructions = vae.decode(latent)[..., : audio.shape[-1]]

        for path, reference, reconstruction, real_len in zip(
            data["paths"], audio, reconstructions, data["real_len"]
        ):
            valid_length = min(
                int(real_len), reference.shape[-1], reconstruction.shape[-1]
            )
            reference = reference[..., :valid_length].unsqueeze(0).float()
            reconstruction = reconstruction[..., :valid_length].unsqueeze(0).float()

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
            values["sisdr"].append(
                -distances["sisdr"](reference_signal, reconstruction_signal).detach().cpu()
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
    print(f"SI-SDR (dB): {results['sisdr']:.4f}")
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
        choices=["KVAE-Audio"],
        default="KVAE-Audio",
        help="Model to load from Hugging Face",
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
    model_paths = {"KVAE-Audio": "kandinskylab/KVAE-Audio"}

    vae = (
        KVAEAudio.from_pretrained(model_paths[cli_args.model])
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
