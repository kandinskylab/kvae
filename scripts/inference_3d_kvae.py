import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional, Literal
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse

from torchmetrics import MetricCollection
from utils.video_metrics import VideoPSNR, VideoSSIM, VideoLPIPS

from kvae.models import KVAE3D

from utils.video_dataset import VideoDataset

from utils.saving_reconstruction_utils import (
    save_results_as_png_async,
    quant_renormalization,
)
from utils.common_utils import set_seed_and_optimal_cuda_env


def run_inference(
    vae: nn.Module,
    device: torch.device,
    data_dir: str,
    batch_size: int = 1,
    saving_folder: Optional[str] = None,
    input_norm: Literal['-11', 'm11', '01'] = 'm11',
    seg_len: Literal[4, 8, 16] = 16
):
    """
    function performs inference using a variational autoencoder on videos, computes
    metrics such as PSNR and LPIPS, and optionally saves reconstructions as PNG files if you need
    
    Parameters
    ----------
    vae: 
        Variational Autoencoder (VAE) model (nn.Module) for inference process
    device: 
        device on which the computations will be performed, such as CPU or GPU. 
        it is of type `torch.device`
    data_dir: 
        directory where the video data is located
    batch_size: 
        number of samples in batch (>1 if our data has the same spatial size and length according to the number of frames)
        a larger batch size can lead to faster processing but may require more memory, defaults to 1
    saving_folder: 
        folder where the reconstructions of the videos will be saved as PNG files as frames
    input_norm:
        type of range of values of input tensor
            ('-11': [-1,1], 'm11': [-1, 127/128], '01': [0,1])
    seg_len:
        length of the sequence processed per iteration
    """
    
    # # for usual video
    # dataset = VideoDataset(data_dir, regex='*', input_norm=cli_args.input_norm)
    # for video, which is folder of .png frames
    dataset = VideoDataset(data_dir, regex='*', stream_pattern='*.png', input_norm=cli_args.input_norm)
    # # for yuv video (we must specify shape, because this is raw format)
    # dataset = VideoDataset(
    #     data_dir, regex="*", input_norm=input_norm, shape=(1280, 720)
    # )

    if len(dataset) == 0:
        raise ValueError(f"No video found in {data_dir}")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
    )

    metrics = MetricCollection(
        {
            "psnr": VideoPSNR(data_range=(-1, 1), metric_chank_size=10),
            "ssim": VideoSSIM(data_range=(-1, 1), metric_chank_size=10),
            "lpips": VideoLPIPS(net_type="alex", metric_chank_size=10),
        }
    ).to(device=device)

    if saving_folder:
        Path(saving_folder).mkdir(parents=True, exist_ok=True)
        save_executor = ThreadPoolExecutor(max_workers=None)

    print(f"Starting inference on {len(dataset)} videos...")

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Processing")):
            videos = data["frames"].to(dtype=dtype, device=device)
            latent = vae.encode(videos, seg_len=seg_len).latent_dist.mode()
            reconstructions = vae.decode(latent, seg_len=seg_len).clip(-1, 1)

            # Saving reconstructions
            if saving_folder:
                for path, rec, real_len in zip(
                    data["paths"], reconstructions, data["real_len"]
                ):
                    save_results_as_png_async(
                        rec[:real_len, ...],
                        Path(saving_folder) / Path(path).stem,
                        save_executor,
                        input_norm=input_norm,
                    )

            # Compute Metrics
            videos = quant_renormalization(
                videos.float(), input_norm=input_norm, output_norm="-11"
            )
            reconstructions = quant_renormalization(
                reconstructions.float(),
                input_norm=input_norm,
                output_norm="-11",
            )

            for video, recon_video, real_len in zip(
                videos, reconstructions, data["real_len"]
            ):
                metrics.update(video[:real_len], recon_video[:real_len])

    results = metrics.compute()

    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"PSNR (dB): {results['psnr_dataset_mean'].item():.4f}")
    print(f"SSIM     : {results['ssim_dataset_mean'].item():.4f}")
    print(f"LPIPS    : {results['lpips_dataset_mean'].item():.4f}")
    print("=" * 40)

    return results["psnr_dataset_mean"].item(), results["lpips_dataset_mean"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KVAE 3D inference")
    parser.add_argument("--device", type=int, default=0, help="GPU number")

    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="./assets/test1/",
        help="Path to image dataset for inference",
    )

    parser.add_argument(
        "--model",
        choices=["KVAE_1.0", "KVAE_2.0-t4s8", "KVAE_2.0-t4s16"],
        default="KVAE_1.0",
        help="Model path for inference",
    )
    parser.add_argument(
        "--input_norm",
        choices=["01", "-11", "m11"],
        default="m11",
        help="Normalizing for input data",
    )

    parser.add_argument(
        "--seg_len",
        type=int,
        choices=[4, 8, 16],
        default=16,
        help="Time length of segment, which we will convert in one pass",
    )
    parser.add_argument(
        "--saving_folder",
        type=str,
        default=None,
        help="Path to folder for saving reconstraction if you need dumping",
    )

    cli_args = parser.parse_args()

    set_seed_and_optimal_cuda_env(111)

    device = torch.device(f"cuda:{cli_args.device}")
    dtype = torch.bfloat16

    model_paths = {
        "KVAE_1.0": "kandinskylab/KVAE-3D-1.0",
        "KVAE_2.0-t4s8": "kandinskylab/KVAE-3D-2.0-t4s8",
        "KVAE_2.0-t4s16": "kandinskylab/KVAE-3D-2.0-t4s16",
    }

    vae = (
        KVAE3D.from_pretrained(model_paths[cli_args.model]).eval().to(device).to(dtype)
    )

    psnr, lpips = run_inference(
        vae=vae,
        data_dir=cli_args.dataset_folder,
        device=device,
        saving_folder=cli_args.saving_folder,
        input_norm=cli_args.input_norm,
        seg_len=cli_args.seg_len
    )
