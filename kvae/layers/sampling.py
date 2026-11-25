from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .conv import SafeConv3d, CachedCausalConv3d


# ==================================================
# =================== 2D Upsample ==================
# ==================================================


class PXSUpsample(nn.Module):
    def __init__(self, in_channels: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            padding_mode="reflect",
        )

        self.linear = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        repeated = x.repeat_interleave(self.factor**2, dim=1)
        pxs_interm = self.shuffle(repeated)

        image_like_ups = F.interpolate(x, scale_factor=2, mode="nearest")
        conv_out = self.spatial_conv(image_like_ups)

        out = conv_out + pxs_interm
        return self.linear(out)


# ==================================================
# =================== 3D Upsample ==================
# ==================================================


UPSAMPLE_VERSION_OPTIONS = {
    1: {"halve_channels": False},
    2: {"halve_channels": True},
}


def get_upsample_version_options(version: int) -> dict:
    if version not in UPSAMPLE_VERSION_OPTIONS:
        raise ValueError(f"Unsupported upsample version: {version}; expected 1 or 2")
    return UPSAMPLE_VERSION_OPTIONS[version]


class CachedPXSUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        compress_time: bool,
        factor: int = 2,
        padding_mode: Literal["zeros", None] = None,
        *,
        version: int = 1,
        halve_channels: Optional[bool] = None,
    ):
        super().__init__()
        version_options = get_upsample_version_options(version)
        self.temporal_compress = compress_time
        self.factor = factor
        self.version = version
        self.halve_channels = (
            version_options["halve_channels"] if halve_channels is None else halve_channels
        )
        if self.halve_channels and in_channels % 2:
            raise ValueError("Channel halving requires an even number of input channels")
        out_channels = in_channels // 2 if self.halve_channels else in_channels
        self.out_channels = out_channels
        
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = SafeConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode=padding_mode or 'reflect',
        )

        if self.temporal_compress:
            self.temporal_conv = CachedCausalConv3d(
                in_channels,
                in_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
                padding_mode=padding_mode
            )

        self.linear = SafeConv3d(out_channels, out_channels, kernel_size=1, stride=1)

    def temporal_upsample(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        # input_ : (T + 1) x H x W
        repeated = input_.repeat_interleave(2, dim=2)

        if self.temporal_conv.cache is None:
            tail = repeated[..., 1:, :, :]
        else:
            tail = repeated

        conv_out = self.temporal_conv(tail)
        return conv_out + tail

    def _spatial_skip_halved(self, input_: torch.Tensor) -> torch.Tensor:
        image_like = rearrange(input_, "b c t h w -> (b t) c h w")
        repeated = image_like.repeat_interleave(self.factor**2 // 2, dim=1)
        return rearrange(
            self.shuffle(repeated), "(b t) c h w -> b c t h w", t=input_.size(2)
        )

    def spatial_upsample(self, input_: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = input_.shape
        input_view = input_.view(b, c * t, h, w)

        input_interp = F.interpolate(input_view, scale_factor=2, mode="nearest")
        input_interp = input_interp.view(b, c, t, 2 * h, 2 * w)
        conv_out = input_interp.new_empty((b, self.out_channels, t, 2 * h, 2 * w))
        conv_out = self.spatial_conv(input_interp, write_to=conv_out)

        if self.halve_channels:
            del input_interp
            return conv_out.add_(self._spatial_skip_halved(input_))
        return conv_out.add_(input_interp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_compress:
            x = self.temporal_upsample(x)

        s_out = self.spatial_upsample(x)
        to = torch.empty_like(s_out)

        lin_out = self.linear(s_out, write_to=to)

        return lin_out


# ==================================================
# ================= 2D Downsample ==================
# ==================================================


class PXSDownsample(nn.Module):
    def __init__(self, in_channels: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.spatial_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            padding_mode="reflect",
        )
        self.linear = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x C x H x W
        pxs_interm = self.unshuffle(x)

        b, c, h, w = pxs_interm.shape
        pxs_interm_view = pxs_interm.view(b, c // self.factor**2, self.factor**2, h, w)
        pxs_out = torch.mean(pxs_interm_view, dim=2)

        conv_out = self.spatial_conv(x)

        out = conv_out + pxs_out
        return self.linear(out)


# ==================================================
# ================= 3D Downsample ==================
# ==================================================


# Temporal entries are (kernel_size_t, stride_t); v1 and v3 share the same operator.
DOWNSAMPLE_VERSION_OPTIONS = {
    1: {"temporal": ((3, 2),), "double_channels": False},
    2: {"temporal": ((2, 1), (2, 2)), "double_channels": True},
    3: {"temporal": ((3, 2),), "double_channels": True},
}


def get_downsample_version_options(version: int) -> dict:
    if version not in DOWNSAMPLE_VERSION_OPTIONS:
        raise ValueError(f"Unsupported downsample version: {version}; expected 1, 2 or 3")
    return DOWNSAMPLE_VERSION_OPTIONS[version]


class CachedPXSDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        compress_time: bool,
        factor: int = 2,
        version: int = 1,
        padding_mode: Literal["zeros", None] = None,
        *,
        double_channels: Optional[bool] = None,
    ):
        super().__init__()
        version_options = get_downsample_version_options(version)
        self.temporal_compress = compress_time
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.s_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        self.version = version
        self.double_channels = (
            version_options["double_channels"] if double_channels is None else double_channels
        )
        out_channels = in_channels * 2 if self.double_channels else in_channels
        self.out_channels = out_channels

        self.spatial_conv = SafeConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            padding_mode=padding_mode or 'reflect',
        )

        if self.temporal_compress:
            temporal_convs = [
                CachedCausalConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=(kernel, 1, 1),
                    stride=(stride, 1, 1),
                    dilation=(1, 1, 1),
                    padding_mode=padding_mode,
                )
                for kernel, stride in version_options["temporal"]
            ]
            # A single conv stays directly under temporal_conv, preserving weight names.
            self.temporal_conv = (
                temporal_convs[0] if len(temporal_convs) == 1 else nn.Sequential(*temporal_convs)
            )

        self.linear = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1)

    def spatial_downsample(self, input_: torch.Tensor) -> torch.Tensor:
        pxs_input = rearrange(input_, "b c t h w -> (b t) c h w")
        pxs_interm = self.unshuffle(pxs_input)
        b, c, h, w = pxs_interm.shape
        if self.double_channels:
            pxs_interm_view = pxs_interm.view(b, c // self.factor, self.factor, h, w)
        else:
            pxs_interm_view = pxs_interm.view(
                b, c // self.factor**2, self.factor**2, h, w
            )
        pxs_out = torch.mean(pxs_interm_view, dim=2)
        pxs_out = rearrange(pxs_out, "(b t) c h w -> b c t h w", t=input_.size(2))

        conv_out = self.spatial_conv(input_)

        return conv_out + pxs_out

    def temporal_downsample(self, input_: torch.Tensor) -> torch.Tensor:
        # Interpolation part
        permuted = rearrange(input_, "b c t h w -> (b h w) c t")

        first_conv = self.temporal_conv[0] if self.version == 2 else self.temporal_conv
        # Read the cache before convolution: only the first chunk keeps its first frame separately.
        if first_conv.cache is None:
            first, rest = permuted[..., :1], permuted[..., 1:]

            if rest.size(-1) > 0:
                rest_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)
                full_interp = torch.cat([first, rest_interp], dim=-1)
            else:
                full_interp = first
        else:
            rest = permuted
            if rest.size(-1) > 0:
                full_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)

        full_interp = rearrange(
            full_interp,
            "(b h w) c t -> b c t h w",
            h=input_.size(-2),
            w=input_.size(-1),
        )

        # Downsampling by convolution
        conv_out = self.temporal_conv(input_)

        return conv_out + full_interp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spatial_downsample(x)

        if self.temporal_compress:
            out = self.temporal_downsample(out)

        return self.linear(out)
