from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor

from .video_stream import Stream


def _norm_to_tensor_from_255(video_np, input_norm="-11"):
    if input_norm == "01":
        video_np = video_np / 255.0
    elif input_norm == "m11":
        video_np = video_np / 128.0 - 1
    elif input_norm == "-11":
        video_np = video_np / 127.5 - 1
    else:
        raise NotImplementedError("Norm type %s is not supported" % input_norm)
    return to_tensor(video_np)


class TruncVideo:
    """
    The `TruncVideo` class truncates the spatial and temporal dimensions
    of a video tensor based on specified compression factors.

    Parameters
    ----------
    spatial_compression:
        specify the compression factor for truncating the spatial dimensions of a video tensor
    temporal_compression:
        specify the compression factor for truncating the temporal dimensions of a video tensor
    """

    def __init__(
        self, spatial_compression: int, temporal_compression: Optional[int]
    ) -> None:
        self.s_factor = spatial_compression
        self.t_factor = temporal_compression

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        h, w = video.shape[-2:]
        new_h = h // self.s_factor * self.s_factor
        new_w = w // self.s_factor * self.s_factor
        video = video[..., :new_h, :new_w]

        if self.t_factor is not None:
            t = video.shape[1]
            new_t = (t - 1) // self.t_factor * self.t_factor + 1
            video = video[:, :new_t, :, :]
        return video


class VideoDataset(Dataset):
    """
    Dataset for loading video data with various options for preprocessing and transformation.
    """

    def __init__(
        self,
        source,
        first_n_frames: int = None,
        regex: str = "*/*",  # data depth 2
        stream_pattern: str = "*",
        transform=None,
        input_norm: str = "-11",
        shape: Tuple[int, int] = None,
    ):
        """
        Parameters
        ----------
        first_n_frames:
            number of frames to consider from the beginning of each video
            if this parameter is set to a specific integer value, only the first `n` frames 
            of each video will be used for processing or analysis
        transform:
            transformations that will be applied to the input data
            if the parameter is not specified, then the TruncVideo(16, 8) is used
        stream_pattern:
            string that specifies the pattern to match the video frames in a directory.
            in this case, the default value for `stream_pattern` is`'*.png'`, which means that the code is expecting
            video frames in PNG format
        input_norm:
            type of input normalization to be applied
        regex:
            pattern for matching videos to process
        """
        if isinstance(source, Path):
            self.source = source
        else:
            self.source = Path(source)

        if self.source.is_dir():
            self.video_paths = sorted(self.source.glob(regex))
        else:
            raise TypeError("We are expecting to receive a folder")

        self.first_n_frames = first_n_frames
        self.stream_pattern = stream_pattern
        self.transform = transform
        self.input_norm = input_norm
        self.shape = shape
        if self.transform is None:
            self.transform = Compose([TruncVideo(16, 8)])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, item):
        path = self.video_paths[item]
        stream = Stream(path, pattern=self.stream_pattern, shape=self.shape)

        length = int(stream.length)
        if self.first_n_frames is not None and self.first_n_frames > 0:
            length = min(self.first_n_frames, stream.length)

        frames = [stream[i] for i in range(length)]
        frames = [
            _norm_to_tensor_from_255(frame, input_norm=self.input_norm)
            for frame in frames
        ]
        frames = torch.stack(frames, dim=1)

        if self.transform:
            frames = self.transform(frames)

        return {
            "paths": str(path),
            "frames": frames,
            "names": path.stem,
            "items": item,
            "real_len": length,
        }


class VideoReader:
    """
    The `VideoReader` class in Python is designed to read and process video frames based on
    specified parameters such as frames per video, sequential frames, normalization, and transformation.
    """

    def __init__(
        self,
        first_n_frames: int = None,
        transform=None,
        stream_pattern: str = "*.png",
        input_norm: str = "-11",
    ):
        """
        Parameters
        ----------
        first_n_frames:
            number of frames to consider from the beginning of each video
            if this parameter is set to a specific integer value, only the first `n` frames of each video will be used
            for processing or analysis
        transform:
            transformations that will be applied to the input data
            if the parameter is not specified, then the TruncVideo(16, 8) is used
        stream_pattern:
            string that specifies the pattern to match the video frames in a directory.
            in this case, the default value for `stream_pattern` is`'*.png'`, which means that the code is expecting
            video frames in PNG format
        input_norm:
            type of input normalization to be applied
        """
        self.first_n_frames = first_n_frames
        self.input_norm = input_norm
        self.transform = transform
        if self.transform is None:
            self.transform = Compose([TruncVideo(16, 8)])
        self.stream_pattern = stream_pattern

    def read_video(self, path) -> torch.Tensor:
        """
        frames from a video file of different formats, processes them, and returns a torch tensor.

        Parameters
        ----------
        path:
            file path to the video that you want to read and process
            it is used to create a `Stream` object to read the video frames from the specified path
        """
        stream = Stream(path, pattern=self.stream_pattern)

        length = int(stream.length)
        if self.first_n_frames is not None and self.first_n_frames > 0:
            length = min(self.first_n_frames, stream.length)

        frames = [stream[i] for i in range(length)]
        frames = [
            _norm_to_tensor_from_255(frame, input_norm=self.input_norm)
            for frame in frames
        ]
        frames = torch.stack(frames, dim=1)

        if self.transform:
            frames = self.transform(frames)

        return {"frames": frames, "real_len": length}
