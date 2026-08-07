from pathlib import Path
from typing import Callable, Tuple, Union

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor


class ImageDataset(Dataset):
    """
    Dataset for loading and processing images.
    """
    def __init__(
        self,
        root_path,
        regex: str = "*.png",
        output_shape: Tuple[int, int] = None,
        transform: Callable = None,
        fake_time_dim: bool = False,
        subset: slice = None,
        output_key: str = "frames",
        name: str = None,
    ):
        """
        Simple image dataset.

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
            if the parameter is not specified, then the Resize(size=output_shape) is used
        output_shape:
            If not None, creates default transform with Resize and ToTensor. Default is None.
            This parameter is ignored if transform is passed.
        output_key:
            Image output key.
        """
        self.root_path = Path(root_path)
        self.image_paths = np.array(sorted(self.root_path.glob(regex)))
        default_transform = (
            Compose([ToTensor(), Resize(size=output_shape)])
            if output_shape is not None
            else ToTensor()
        )
        self.transform = transform or default_transform
        self.output_key = output_key
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
        if self.fake_time:
            frames = frames.unsqueeze(1)

        output_dict = {"paths": path, self.output_key: frames}

        return output_dict


def read_image(path: Union[Path, str]):
    frames = cv2.imread(str(path))
    frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    frames = ToTensor()(frames)
    frames = 2 * frames - 1
    return frames