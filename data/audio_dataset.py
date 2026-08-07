from pathlib import Path
from typing import Callable, Optional, Union

import torch
from audiotools import AudioSignal
from audiotools.core import util
from torch.utils.data import Dataset


def read_audio(path: Union[str, Path]) -> AudioSignal:
    """Read an audio file using audiotools.

    Parameters
    ----------
    path:
        Path to an audio file.

    Returns
    -------
    AudioSignal
        Loaded signal with audio data shaped as [b, c, s]
        and the original sample rate.
    """
    return AudioSignal(Path(path))


class AudioDataset(Dataset):
    """Dataset for loading audio files through audiotools.AudioSignal.

    Parameters
    ----------
    root_path:
        Path to an audio file or a directory. Directories are searched
        recursively using audiotools.core.util.find_audio.
    transform:
        Optional transformation applied to an audio tensor shaped as
        [channels, samples].
    subset:
        Optional slice selecting a subset of discovered audio files.
    output_key:
        Key used for the audio tensor in the returned dictionary.
    name:
        Optional dataset name. By default, the root path name is used.

    Notes
    -----
    The dataset does not resample, downmix, crop, pad, or normalize audio.
    An item contains an unbatched [channels, samples] tensor. DataLoader with
    batch_size=1 produces the [batch, channels, samples] shape. 
    Variable-length files require batch_size=1 or a custom collate function.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        subset: Optional[slice] = None,
        output_key: str = "audio",
        name: Optional[str] = None,
    ) -> None:
        self.root_path = Path(root_path)
        self.audio_paths = sorted(
            Path(path) for path in util.find_audio(self.root_path)
        )
        self.transform = transform
        self.output_key = output_key

        if subset is not None:
            self.audio_paths = self.audio_paths[subset]

        self.name = name or self.root_path.name

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, item: int) -> dict:
        path = self.audio_paths[item]
        signal = read_audio(path)
        audio = signal.audio_data[0]

        if self.transform is not None:
            audio = self.transform(audio)

        return {
            "paths": str(path),
            self.output_key: audio,
            "sample_rate": signal.sample_rate,
            "names": path.stem,
            "items": item,
            "real_len": audio.shape[-1],
        }
