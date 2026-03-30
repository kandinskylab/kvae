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


class CachedPXSUpsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = SafeConv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="reflect",
        )

        if self.temporal_compress:
            self.temporal_conv = CachedCausalConv3d(
                in_channels,
                in_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
            )

        self.linear = SafeConv3d(in_channels, in_channels, kernel_size=1, stride=1)

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

    def spatial_upsample(self, input_: torch.Tensor) -> torch.Tensor:
        def conv_part(x):
            to = torch.empty_like(x)
            out = self.spatial_conv(x, write_to=to)
            return out

        b, t, c, h, w = input_.shape
        input_view = input_.view(b, (t * c), h, w)

        input_interp = F.interpolate(input_view, scale_factor=2, mode="nearest")
        input_interp = input_interp.view(b, t, c, 2 * h, 2 * w)
        input_interp.add_(conv_part(input_interp))
        return input_interp

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


class CachedPXSDownsample(nn.Module):
    def __init__(
        self, in_channels: int, compress_time: bool, factor: int = 2, version: int = 1
    ):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.s_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        self.version = version
        out_channels = in_channels * 2 if version > 1 else in_channels

        self.spatial_conv = SafeConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            padding_mode="reflect",
        )

        if self.temporal_compress:
            if version == 2:
                self.temporal_conv = nn.Sequential(
                    CachedCausalConv3d(
                        out_channels,
                        out_channels,
                        kernel_size=(2, 1, 1),
                        stride=(1, 1, 1),
                        dilation=(1, 1, 1),
                    ),
                    CachedCausalConv3d(
                        out_channels,
                        out_channels,
                        kernel_size=(2, 1, 1),
                        stride=(2, 1, 1),
                        dilation=(1, 1, 1),
                    ),
                )
            else:
                self.temporal_conv = CachedCausalConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=(3, 1, 1),
                    stride=(2, 1, 1),
                    dilation=(1, 1, 1),
                )

        self.linear = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1)

    def spatial_downsample(self, input_: torch.Tensor) -> torch.Tensor:
        pxs_input = rearrange(input_, "b c t h w -> (b t) c h w")
        pxs_interm = self.unshuffle(pxs_input)
        b, c, h, w = pxs_interm.shape
        if self.version > 1:
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

        cached_flag = (
            self.temporal_conv.cache
            if self.version == 1
            else self.temporal_conv[0].cache
        )
        if cached_flag is None:
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
        if self.version == 1:
            conv_out = self.temporal_conv(input_)
        elif self.version == 2:
            conv_out = self.temporal_conv[0](input_)
            conv_out = self.temporal_conv[1](conv_out)

        return conv_out + full_interp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spatial_downsample(x)

        if self.temporal_compress:
            out = self.temporal_downsample(out)

        return self.linear(out)
