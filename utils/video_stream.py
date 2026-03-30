import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class Stream:
    """
    Video stream class. Access video as a file or as a directory with frames.
    """

    def __init__(
        self, path: str | Path, shape: Tuple[int, int] = None, pattern: str = "*.png"
    ):
        """
        Parameters
        ----------
        path:
            Path to video sample or directory with frames.
        shape:
            Frame WxH shape. Expected if input is YUV video.
        pattern:
            Pattern to look for in file names in the directory.
        """
        self.path = Path(path)
        if shape is not None:
            self.width, self.height = shape
        if not self.path.exists():
            raise FileExistsError(
                f"File or directory '{str(self.path)}' does not exist."
            )
        if self.path.is_file():
            if self.path.suffix.lower() == ".yuv":
                self.input_mode = "raw"
            else:
                self.input_mode = "file"
        else:
            self.input_mode = "folder"
        self._stream = None
        self.frame_paths: List[Path] = []
        self.pattern = pattern
        self.reset()
        self._update_info()

    def reset(self):
        """
        Resets stream to first frame.
        """
        if self.input_mode == "file":
            if self._stream is not None:
                self._stream.release()
            self._stream = cv2.VideoCapture(str(self.path))
            if not self._stream.isOpened():
                raise IOError(f"Unable to open stream for '{self.path}'")
        elif self.input_mode == "raw":
            if self._stream is not None:
                self._stream.close()
            self._stream = open(self.path, "rb")
        elif self.input_mode == "folder":
            self.frame_paths = sorted(self.path.glob(self.pattern))
        else:
            raise ValueError("Unknown input_mode")

    def _update_info(self):
        if self.input_mode == "file":
            self.width = int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.length = int(self._stream.get(cv2.CAP_PROP_FRAME_COUNT))
        elif self.input_mode == "raw":
            file_size = os.path.getsize(self.path)
            self.length = file_size // (self.width * self.height * 3 // 2)
        else:
            self.width, self.height = cv2.imread(str(self.frame_paths[0])).shape[:2][
                ::-1
            ]
            self.length = len(self.frame_paths)
        self.shape = (self.length, self.height, self.width, 3)

    def _convert_color(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __iter__(self):
        """
        Returns iterator over frames.
        """
        self.reset()
        if self.input_mode == "file":
            while self._stream.isOpened():
                ret, frame = self._stream.read()
                if not ret:
                    break
                yield self._convert_color(frame)
            self._stream.release()
        if self.input_mode == "raw":
            for _ in range(self.length):
                yuv = np.frombuffer(
                    self._stream.read(self.width * self.height * 3 // 2), dtype=np.uint8
                ).reshape((self.height * 3 // 2, self.width))
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                yield self._convert_color(bgr)
        else:
            for path in self.frame_paths:
                frame = cv2.imread(str(path))
                yield self._convert_color(frame)

    def __getitem__(self, item: int):
        """
        Returns frame by id.
        """
        if not 0 <= item < self.length:
            raise IndexError("Index out of sequence length")
        if self.input_mode == "file":
            self._stream.set(cv2.CAP_PROP_POS_FRAMES, item)
            ret, frame = self._stream.read()
            if not ret:
                raise ValueError(f"Unable to read frame {item}")
        elif self.input_mode == "raw":
            self.reset()
            for _ in range(item):
                self._stream.read(self.width * self.height * 3 // 2)

            yuv = np.frombuffer(
                self._stream.read(self.width * self.height * 3 // 2), dtype=np.uint8
            ).reshape((self.height * 3 // 2, self.width))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        else:
            frame = cv2.imread(str(self.frame_paths[item]))
        return self._convert_color(frame)

    def __len__(self):
        """
        Returns number of frames of the video sample.
        """
        return self.length
