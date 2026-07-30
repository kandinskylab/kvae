from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, Optional, Literal

import numpy as np
import torch
from PIL import Image


def _save_single_frame(frame, output_path, frame_index):
    img = Image.fromarray(frame)
    img.save(output_path / f"frame_{frame_index:03d}.png")


def _norm_to_255numpy(tensor, input_norm="m11"):
    if input_norm == "01":
        return (tensor.clip(0, 1).numpy(force=True) * 255).round().astype(np.uint8)
    elif input_norm == "m11":
        return (
            ((tensor.clip(-1, 127 / 128).numpy(force=True) + 1) * 128)
            .round()
            .astype(np.uint8)
        )
    elif input_norm == "-11":
        return (
            ((tensor.clip(-1, 1).numpy(force=True) + 1) * 127.5)
            .round()
            .astype(np.uint8)
        )
    else:
        raise ValueError(f"Unknown input_norm {input_norm}")


def tensor_norm_to_255(video_tensor, input_norm="m11"):
    if input_norm == "01":
        return (video_tensor.clip(0, 1) * 255).round().to(dtype=torch.uint8)
    elif input_norm == "m11":
        return (
            ((video_tensor.clip(-1, 127 / 128) + 1) * 128).round().to(dtype=torch.uint8)
        )
    elif input_norm == "-11":
        return ((video_tensor.clip(-1, 1) + 1) * 127.5).round().to(dtype=torch.uint8)
    else:
        raise ValueError(f"Unknown input_norm {input_norm}")


def tensor_norm_from_255(video_tensor: Union[torch.Tensor, np.array], input_norm="m11"):
    if input_norm == "01":
        return video_tensor / 255.0
    elif input_norm == "m11":
        return video_tensor / 128 - 1.0
    elif input_norm == "-11":
        return video_tensor / 127.5 - 1.0
    else:
        raise ValueError(f"Unknown input_norm {input_norm}")


def quant_renormalization(
    input_tensor: torch.Tensor,
    input_norm: Literal["01", "-11", "m11"] = "-11",
    output_norm: Literal["01", "-11", "m11"] = "-11",
):
    input_tensor.transpose_(1, 2)
    input_tensor = tensor_norm_to_255(input_tensor.float(), input_norm=input_norm)
    input_tensor = tensor_norm_from_255(input_tensor, input_norm=output_norm)
    return input_tensor


def save_results_as_png_async(
    video_tensor: torch.Tensor,
    output_dir: Union[str, Path],
    executor: ThreadPoolExecutor,
    input_norm: Literal["-11", "m11", "01"] = "m11",
):
    """
    The function `save_results_as_png_async` asynchronously saves each frame of a video tensor as a PNG
    image using a ThreadPoolExecutor.

    Parameters
    ----------
    video_tensor:
        input tensor [C, T, H, W] shape
    output_dir:
        path to dir for saving frames of result video
    executor:
        the executor that the save goes through
    input_norm:
        type of range of values of input tensor
            ('-11': [-1,1], 'm11': [-1, 127/128], '01': [0,1])
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_tensor = _norm_to_255numpy(
        video_tensor.cpu().float().permute(1, 2, 3, 0), input_norm=input_norm
    )
    futures = []
    # This code snippet is iterating over each frame in the `video_tensor` and submitting a task to
    # the `executor` to asynchronously save each frame as a PNG image.
    for i, frame in enumerate(video_tensor):
        future = executor.submit(_save_single_frame, frame, output_dir, i)
        futures.append(future)


def save_tensor_image(
    tensor: torch.Tensor,
    save_dir_path: Union[str, Path],
    filename: str,
    denormalize_fn: Optional[callable] = _norm_to_255numpy,
    input_norm: Literal["-11", "m11", "01"] = "-11",
) -> Path:
    """
    This function saves a torch tensor as an image file after denormalizing and converting it to a
    suitable format.

    Parameters
    ----------
    tensor:
        input tensor [C, H, W] shape
    save_dir_path:
        path to dir for saving result
    filename:
        name of saving image
    denormalize_fn:
        function for denormalisation tensor to uint8 format
    input_norm:
        type of range of values of input tensor
            ('-11': [-1,1], 'm11': [-1, 127/128], '01': [0,1])
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"A torch.Tensor is expected, received input of the type {type(tensor)}"
        )

    if tensor.dim() != 3:
        raise ValueError(
            f"A torch.Tensor of the shape [C, H, W] is expected, received shape {tensor.shape}"
        )

    save_dir_path = Path(save_dir_path)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    if "." not in filename:
        filename = f"{filename}.png"

    file_path = save_dir_path / filename

    img = denormalize_fn(tensor, input_norm=input_norm)
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC

    pil_img = Image.fromarray(img)
    pil_img.save(file_path)

    # print(f"Image saved: {file_path} ({file_path.stat().st_size / 1024:.2f} KB)")

    return file_path
