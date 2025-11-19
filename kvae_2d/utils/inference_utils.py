from pathlib import Path
from typing import Union, List, Iterable, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from .data import ImageDataset
from .metrics_utils import compute_psnr_range, lpips_score, add_fid_images


def recursive_to(obj: Iterable[Any], device: Union[str, torch.device], dtype: torch.dtype = None):
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = recursive_to(obj[i], device=device, dtype=dtype)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = recursive_to(value, device=device, dtype=dtype)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        if dtype is not None:
            obj = obj.type(dtype)
        return obj
    return obj


def mean_value(values: List[Union[torch.Tensor, int, float]]) -> np.ndarray:
    if isinstance(values[0], (int, float)):
        return np.mean(values)
    elif isinstance(values[0], torch.Tensor):
        values = torch.tensor([v.mean() for v in values])
        return torch.mean(values).cpu().float().numpy()
    raise TypeError


def save_image(save_path, image: torch.Tensor, scale_zero_one: bool = False):
    image = image.detach()
    if not scale_zero_one:
        image = (image + 1) / 2

    if image.ndim == 5 and image.shape[0] == 1 and image.shape[2] == 1:
        image = image[0, :, 0, :, :]  # video-like image
    elif image.ndim == 4 and image.shape[0] == 1:
        image = image[0]  # batched
    if image.ndim != 3:
        raise ValueError("Expected input with shape [C, H, W], [1, C, H, W] or [1, C, 1, H, W]")
    if image.shape[0] == 1:  # gray scale
        image = image[0]
    else:
        image = image.permute([1, 2, 0])
    image = image.detach().cpu().numpy()
    image = (image.clip(0, 1) * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@torch.inference_mode()
def test_dataset(
    dataset,
    model,
    save_path: Path = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    scale_zero_one: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
):
    if save_path is not None:
        save_path.mkdir(exist_ok=True, parents=True)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=scale_zero_one).eval().to(device).to(dtype)
    fid = FrechetInceptionDistance(normalize=True).eval().to(device)  # mb: float64
    psnrs = []
    l1s = []
    lpips_losses = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch in tqdm(dataloader, leave=False, desc=f"Validation on {dataset.name}"):
        batch["frames"] = batch["frames"].to(dtype).to(device)
        output = model.encode(batch["frames"])
        output["x_hat"] = model.decode(output["y_hat"])

        per_frame_lpips = lpips_score(output["x_hat"], batch["frames"], lpips=lpips, scale_zero_one=scale_zero_one)
        add_fid_images(fid, batch["frames"], real=True)
        add_fid_images(fid, output["x_hat"], real=False)

        per_frame_psnr = compute_psnr_range(batch["frames"], output["x_hat"], input_type="image")
        per_frame_l1 = F.l1_loss(batch["frames"], output["x_hat"])

        psnrs.append(per_frame_psnr.item())
        l1s.append(per_frame_l1.item())
        lpips_losses.append(per_frame_lpips.item())

        output = recursive_to(output, "cpu", torch.float)

        if save_path is not None:
            for i, path in enumerate(batch["paths"]):
                image_save_path = save_path / f"{Path(path).stem}.png"
                save_image(image_save_path, output["x_hat"][i].squeeze(), scale_zero_one=scale_zero_one)

    fid_score = fid.compute().cpu().numpy()

    return {
        "psnr": psnrs,
        "l1": l1s,
        "lpips": lpips_losses,
        "fid": fid_score,
    }


@torch.inference_mode()
def validate(
    model,
    save_path: str = None,
    image_save_path=None,
    datasets: List[Union[str, ImageDataset]] = None,
    loss: nn.Module = None,
    device: Union[str, torch.device] = "cpu",
    loss_device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = None,
    num_workers: int = 0,
    eq_scale: int = None,
    eq_mode: str = "bicubic",
    reevaluate: bool = False,
):
    save_path = Path(save_path) if save_path is not None else None
    image_save_path = Path(image_save_path) if image_save_path else None

    if save_path is not None and save_path.exists():
        accumulated_results = torch.load(save_path, weights_only=False)
    else:
        accumulated_results = {}

    for dataset in datasets:
        if dataset.name in accumulated_results and not reevaluate:
            continue
        accumulated_results[dataset.name] = {}
        results = test_dataset(
            dataset=dataset,
            model=model,
            loss=loss,
            save_path=image_save_path / dataset.name if image_save_path else None,
            device=device,
            loss_device=loss_device,
            dtype=dtype,
            batch_size=batch_size or getattr(dataset, "batch_size", 1),
            num_workers=num_workers,
            scale=eq_scale,
            eq_mode=eq_mode,
        )
        print(f"{dataset.name}|", end=" ")
        for key in results.keys():
            value = np.mean(results[key])
            accumulated_results[dataset.name][key] = value
            print(f"{key}: {value:.4f}", end=" ")
        print()

    if save_path is not None:
        torch.save(accumulated_results, save_path)

    return accumulated_results
