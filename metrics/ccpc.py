from typing import Sequence

import torch

DEFAULT_FFT_SIZES = (512, 1024, 2048, 4096)


def _as_stereo(audio: torch.Tensor) -> torch.Tensor:
    """Treat a mono waveform as stereo with identical channels."""
    if audio.size(-2) == 1:
        return audio.expand(*audio.shape[:-2], 2, audio.shape[-1])
    return audio


def _stft_channel(audio: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
    return torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )


def _ccpc_single_resolution(
    ref_l: torch.Tensor,
    ref_r: torch.Tensor,
    est_l: torch.Tensor,
    est_r: torch.Tensor,
    n_fft: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """CCPC at one STFT resolution. Inputs are real waveforms, shape [B, T]."""
    hop_length = n_fft // 4
    ref_left = _stft_channel(ref_l, n_fft, hop_length)
    ref_right = _stft_channel(ref_r, n_fft, hop_length)
    est_left = _stft_channel(est_l, n_fft, hop_length)
    est_right = _stft_channel(est_r, n_fft, hop_length)

    n_frames = min(ref_left.size(-1), est_left.size(-1))
    ref_left, ref_right = ref_left[..., :n_frames], ref_right[..., :n_frames]
    est_left, est_right = est_left[..., :n_frames], est_right[..., :n_frames]

    ref_cross = ref_left * ref_right.conj()
    est_cross = est_left * est_right.conj()
    ref_mag = ref_cross.abs()
    est_mag = est_cross.abs()
    phasor_err = (ref_cross / ref_mag.clamp_min(eps)) * (est_cross / est_mag.clamp_min(eps)).conj()
    weight = (ref_mag * est_mag).sqrt()
    coherence = (weight * phasor_err).sum(dim=-2).abs()
    energy = weight.sum(dim=-2)
    return coherence.sum(dim=-1) / energy.sum(dim=-1).clamp_min(eps)


@torch.no_grad()
def compute_ccpc(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    fft_sizes: Sequence[int] = DEFAULT_FFT_SIZES,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross-channel phase coherence, averaged over four STFT resolutions.

    Waveforms are `[batch, channels, time]` or `[channels, time]`. A mono
    signal is treated as two identical channels. The score is in `[0, 1]`.
    """
    if reference.dim() == 2:
        reference = reference.unsqueeze(0)
    if estimate.dim() == 2:
        estimate = estimate.unsqueeze(0)
    reference = _as_stereo(reference)
    estimate = _as_stereo(estimate)
    n_samples = min(reference.size(-1), estimate.size(-1))
    reference = reference[..., :n_samples].float()
    estimate = estimate[..., :n_samples].float()
    scores = [
        _ccpc_single_resolution(
            reference[:, 0],
            reference[:, 1],
            estimate[:, 0],
            estimate[:, 1],
            n_fft,
            eps=eps,
        )
        for n_fft in fft_sizes
    ]
    return torch.stack(scores, dim=0).mean(dim=0)
