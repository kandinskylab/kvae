from typing import Callable, Optional, Sequence, Union

import torch
from audiotools import AudioSignal, STFTParams
from torch import nn

__all__ = [
    "WaveformL1Distance",
    "SISDRDistance",
    "MultiScaleSTFTDistance",
    "MelSpectrogramDistance",
]


class WaveformL1Distance(nn.L1Loss):
    """L1 distance between waveform tensors or AudioSignal attributes."""

    def __init__(self, attribute: str = "audio_data", **kwargs):
        super().__init__(**kwargs)
        self.attribute = attribute

    def forward(
        self,
        reference: Union[AudioSignal, torch.Tensor],
        estimate: Union[AudioSignal, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(reference, AudioSignal):
            reference = getattr(reference, self.attribute)
            estimate = getattr(estimate, self.attribute)
        return super().forward(reference, estimate)


class SISDRDistance(nn.Module):
    """Negative SI-SDR distance with the original KVAE-Audio formulation."""

    def __init__(
        self,
        scaling: bool = True,
        reduction: str = "mean",
        zero_mean: bool = True,
        clip_min: Optional[float] = None,
    ):
        super().__init__()
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min

    def forward(
        self,
        reference: Union[AudioSignal, torch.Tensor],
        estimate: Union[AudioSignal, torch.Tensor],
    ) -> torch.Tensor:
        eps = 1e-8
        if isinstance(reference, AudioSignal):
            references = reference.audio_data
            estimates = estimate.audio_data
        else:
            references = reference
            estimates = estimate

        batch_size = references.shape[0]
        references = references.reshape(batch_size, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(batch_size, 1, -1).permute(0, 2, 1)

        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        references = references - mean_reference
        estimates = estimates - mean_estimate

        references_projection = (references**2).sum(dim=-2) + eps
        references_on_estimates = (estimates * references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        target = scale * references
        residual = estimates - target

        signal = (target**2).sum(dim=1)
        noise = (residual**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()

        return sdr


class MultiScaleSTFTDistance(nn.Module):
    """Magnitude and log-magnitude distance over several STFT resolutions."""

    def __init__(
        self,
        window_lengths: Sequence[int] = (2048, 512),
        distance_fn: Optional[Callable] = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        match_stride: bool = False,
        window_type: Optional[str] = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.distance_fn = nn.L1Loss() if distance_fn is None else distance_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.pow = pow

    def forward(
        self,
        reference: AudioSignal,
        estimate: AudioSignal,
    ) -> torch.Tensor:
        distance = 0.0
        for stft_params in self.stft_params:
            reference_i = reference.clone()
            estimate_i = estimate.clone()
            reference_i.stft(
                stft_params.window_length,
                stft_params.hop_length,
                stft_params.window_type,
            )
            estimate_i.stft(
                stft_params.window_length,
                stft_params.hop_length,
                stft_params.window_type,
            )
            distance += self.log_weight * self.distance_fn(
                reference_i.magnitude.clamp(self.clamp_eps)
                .pow(self.pow)
                .log10(),
                estimate_i.magnitude.clamp(self.clamp_eps)
                .pow(self.pow)
                .log10(),
            )
            distance += self.mag_weight * self.distance_fn(
                reference_i.magnitude,
                estimate_i.magnitude,
            )

        return distance


class MelSpectrogramDistance(nn.Module):
    """Magnitude and log-magnitude distance over several mel resolutions."""

    def __init__(
        self,
        n_mels: Sequence[int] = (150, 80),
        window_lengths: Sequence[int] = (2048, 512),
        distance_fn: Optional[Callable] = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        match_stride: bool = False,
        mel_fmin: Sequence[float] = (0.0, 0.0),
        mel_fmax: Sequence[Optional[float]] = (None, None),
        window_type: Optional[str] = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = tuple(n_mels)
        self.distance_fn = nn.L1Loss() if distance_fn is None else distance_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.mel_fmin = tuple(mel_fmin)
        self.mel_fmax = tuple(mel_fmax)
        self.pow = pow

    def forward(
        self,
        reference: AudioSignal,
        estimate: AudioSignal,
    ) -> torch.Tensor:
        distance = 0.0
        for n_mels, fmin, fmax, stft_params in zip(
            self.n_mels,
            self.mel_fmin,
            self.mel_fmax,
            self.stft_params,
        ):
            reference_i = reference.clone()
            estimate_i = estimate.clone()
            kwargs = {
                "window_length": stft_params.window_length,
                "hop_length": stft_params.hop_length,
                "window_type": stft_params.window_type,
            }
            reference_mels = reference_i.mel_spectrogram(
                n_mels,
                mel_fmin=fmin,
                mel_fmax=fmax,
                **kwargs,
            )
            estimate_mels = estimate_i.mel_spectrogram(
                n_mels,
                mel_fmin=fmin,
                mel_fmax=fmax,
                **kwargs,
            )

            distance += self.log_weight * self.distance_fn(
                reference_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                estimate_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            distance += self.mag_weight * self.distance_fn(
                reference_mels,
                estimate_mels,
            )

        return distance
