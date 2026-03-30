import functools
from typing import Optional, Callable

import torch
import torch.nn as nn
from diffusers.models.activations import get_activation

from .common import get_activation_with_kwargs
from .conv import CachedCausalConv3d, SafeConv3d
from .norm import get_normalization


# ==================================================
# =================== 2D Modules ===================
# ==================================================


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        zq_ch: Optional[int] = None,
        normalization: nn.Module = nn.GroupNorm,
        act_fn: str = "swish"
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(act_fn)

        if zq_ch is None:
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
        if zq_ch is None:
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
        act_fn: str = "swish"
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
        )

        self.norm2 = normalization(
            out_channels,
            zq_ch=zq_ch,
        )
        self.conv2 = CachedCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
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
