from pathlib import Path
from typing import Tuple, Callable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize


class ImageDataset(Dataset):
    def __init__(
        self,
        root_path,
        regex: str = "*.png",
        output_shape: Tuple[int, int] = None,
        transform: Callable = None,
        fake_time_dim: bool = False,
        subset: slice = None,
        name: str = None,
    ):
        """Simple image dataset.

        Parameters
        ----------
        root_path:
            Path to the root directory of the dataset.
        regex:
            Regular expression for the files.
        subset:
            Slice of the dataset to use.
        transform:
            Torchvision transforms to apply to the item.
        output_shape:
            If not None, creates default transform with Resize and ToTensor. Default is (256, 256).
            This parameter is ignored if transform is passed.
        """
        self.root_path = Path(root_path)
        self.image_paths = np.array(sorted(self.root_path.glob(regex)))
        default_transform = Compose([ToTensor(), Resize(size=output_shape)]) if output_shape is not None else ToTensor()
        self.transform = transform or default_transform
        self.fake_time = fake_time_dim

        if subset is not None:
            self.image_paths = self.image_paths[subset]

        self.name = name or self.root_path.name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        path = str(self.image_paths[item])
        frames = cv2.imread(path)
        frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
        if self.transform:
            frames = self.transform(frames)
        frames = 2 * frames - 1
        return frames


class CloseCenterCrop:
    def __init__(self, shape_divisor: Sequence[int] = (8, 8)):
        """Crop to nearest shape divisible by `shape_divisor`.

        Parameters
        ----------
        shape_divisor: Tuple[int]
            Divisors of shape. Same dim order as in the image exept first dim (CTHW).
        """
        self.shape_divisor = torch.tensor([1] + list(shape_divisor))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        assert img.ndim == len(
            self.shape_divisor
        ), "input should have exactly one additional first channel dimension to dimensions or shape_divisor"
        shape = torch.tensor(img.shape)
        new_shape = (shape // self.shape_divisor) * self.shape_divisor
        start = (shape - new_shape) // 2
        end = start + new_shape
        sl = tuple(slice(s, e) for s, e in zip(start, end))
        return img[sl].clone()
