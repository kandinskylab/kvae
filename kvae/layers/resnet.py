import functools
from typing import Optional, Callable, Literal

import torch
import torch.nn as nn
from diffusers.models.activations import get_activation

from .common import get_activation_with_kwargs, Snake1d
from .conv import CachedCausalConv3d, SafeConv3d, WNConv1d
from .norm import get_normalization

# ==================================================
# =================== 1D Modules ===================
# ==================================================

class ResnetBlock1D(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


# ==================================================
# =================== 2D Modules ===================
# ==================================================

def _apply_norm_2d(norm: nn.Module, h: torch.Tensor, zq: Optional[torch.Tensor]) -> torch.Tensor:
    if getattr(norm, "requires_zq", False):
        if zq is None:
            raise ValueError(f"{type(norm).__name__} requires zq")
        return norm(h, zq=zq)
    return norm(h)


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        zq_ch: Optional[int] = None,
        normalization: nn.Module = nn.GroupNorm,
        act_fn: str = "swish",
        norm_factory: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(act_fn)

        if norm_factory is not None:
            self.norm1 = norm_factory(in_channels)
        elif zq_ch is None:
            self.norm1 = normalization(num_channels=in_channels, num_groups=32, eps=1e-6, affine=True)
        else:
            self.norm1 = normalization(in_channels, zq_ch=zq_ch)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=(1, 1),
            padding_mode="replicate",
        )
        if norm_factory is not None:
            self.norm2 = norm_factory(out_channels)
        elif zq_ch is None:
            self.norm2 = normalization(num_channels=out_channels, num_groups=32, eps=1e-6, affine=True)
        else:
            self.norm2 = normalization(out_channels, zq_ch=zq_ch)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=(1, 1),
            padding_mode="replicate",
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    @property
    def requires_zq(self) -> bool:
        """Whether either normalization needs the conditioning latent."""
        return any(
            getattr(norm, "requires_zq", False)
            for norm in (self.norm1, self.norm2)
        )

    def forward(self, x: torch.Tensor, zq: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x

        h = _apply_norm_2d(self.norm1, h, zq)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = _apply_norm_2d(self.norm2, h, zq)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


# ==================================================
# =================== 3D Modules ===================
# ==================================================
class CachedCausalResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        zq_ch: Optional[int] = None,
        normalization: Callable = functools.partial(get_normalization, norm_name="group_norm"),
        act_fn: str = "swish",
        padding_mode: Literal["zeros", None] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation_with_kwargs(act_fn, inplace=True)

        self.norm1 = normalization(
            in_channels,
            zq_ch=zq_ch,
        )

        self.conv1 = CachedCausalConv3d(
            chan_in=in_channels,
            chan_out=out_channels,
            kernel_size=3,
            padding_mode=padding_mode,
        )

        self.norm2 = normalization(
            out_channels,
            zq_ch=zq_ch,
        )
        self.conv2 = CachedCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
            padding_mode=padding_mode,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = SafeConv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, x: torch.Tensor, zq: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x

        h = self.norm1(h, zq) if zq is not None else self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h, zq) if zq is not None else self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h
