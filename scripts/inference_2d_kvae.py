import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    LearnedPerceptualImagePatchSimilarity,
)
from tqdm import tqdm

from kvae.models import KVAE2D
from utils.common_utils import parse_int_tuple, set_seed
from utils.image_dataset import ImageDataset
from utils.saving_reconstruction_utils import save_tensor_image


def run_inference(
    vae: nn.Module,
    device: torch.device,
    data_dir: str,
    batch_size: int = 1,
    img_size: Optional[Tuple[int, int]] = None,
    saving_folder: Optional[str] = None,
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
    img_size:
        output shape of the images in the dataset
    saving_folder: 
        folder where the reconstructions of the videos will be saved as PNG files as frames
    """
    if img_size is None:
        batch_size = 1

    dataset = ImageDataset(root_path=data_dir, regex="*.png", output_shape=img_size)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    metrics = MetricCollection(
        {
            "psnr": PeakSignalNoiseRatio(data_range=(-1, 1), reduction="none", dim=[1, 2, 3]),
            "lpips": LearnedPerceptualImagePatchSimilarity(reduction="none", net_type="alex"),
        }
    ).to(device=device)

    print(f"Starting inference on {len(dataset)} images...")

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Processing")):
            images = data["frames"].to(dtype=dtype, device=device)

            latent = vae.encode(images).latent_dist.mode()
            reconstructions = vae.decode(latent).clip(-1, 1)

            # Compute Metrics
            metrics.update(reconstructions.float(), images.float())

            # Saving reconstructions
            if saving_folder:
                for path, rec in zip(data["paths"], reconstructions):
                    save_tensor_image(
                        rec.float().cpu(),
                        save_dir_path=saving_folder,
                        filename=Path(path).name,
                    )

    result = metrics.compute()

    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"PSNR (dB): {result['psnr'].mean().item():.4f}")
    print(f"LPIPS    : {result['lpips'].mean().item():.4f}")
    print("=" * 40)

    return result["psnr"].mean().item(), result["lpips"].mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KVAE 2D inference")
    parser.add_argument("--device", type=int, default=0, help="GPU number")

    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="./assets/images/",
        help="Path to image dataset for inference",
    )
    parser.add_argument("--bs", type=int, default=1, help="Batchsize for Dataloader")
    parser.add_argument(
        "--img_size",
        type=parse_int_tuple,
        default=None,
        help="Size of images 'H,W' for resize for collecting in batch",
    )

    parser.add_argument(
        "--model",
        choices=["KVAE_1.0"],
        default="KVAE_1.0",
        help="Model path for inference",
    )
    parser.add_argument(
        "--saving_folder",
        type=str,
        default=None,
        help="Path to folder for saving reconstraction if you need dumping",
    )

    cli_args = parser.parse_args()

    set_seed(111)

    device = torch.device(f"cuda:{cli_args.device}")
    dtype = torch.bfloat16

    model_paths = {
        "KVAE_1.0": "kandinskylab/KVAE-2D-1.0",
    }

    vae = KVAE2D.from_pretrained(model_paths[cli_args.model]).eval().to(device).to(dtype)

    psnr, lpips = run_inference(
        vae=vae,
        data_dir=cli_args.dataset_folder,
        batch_size=cli_args.bs,
        img_size=cli_args.img_size,
        device=device,
        saving_folder=cli_args.saving_folder
    )
