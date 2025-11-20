import torch
import numpy as np

from PIL import Image


def count_parameters(model: torch.nn.Module) -> int:
    """Counts number of parameters in the model.

    Parameters
    ----------
    model: nn.Module
        the model

    Returns
    -------
    number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_image(image: torch.Tensor, scale_zero_one: bool = False):
    """Displays torch image in Ipython notebook.

    Notes
    -----
    Converts torch image from [0, 1] to [0, 255] format.

    Parameters
    ----------
    image: torch.Tensor
        Image as tensor with dimensions [C, H, W] or [B, C, H, W].
    scale_zero_one: bool, optional
        If true, expects image to be in [0, 1] range and in [-1, 1] otherwise. Default is False.
    """
    from IPython.display import display

    if not scale_zero_one:
        image = (image + 1) / 2

    if isinstance(image, torch.Tensor):
        if image.ndim > 3:  # batched
            image = image[0]
        if image.shape[0] == 1:  # gray scale
            image = image[0]
        else:
            image = image.permute([1, 2, 0])
        image = image.detach().float().cpu().numpy()
        image = (image.clip(0, 1) * 255).astype(np.uint8)
    display(Image.fromarray(image))
