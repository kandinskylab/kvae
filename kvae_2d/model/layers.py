import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):  # swish
    return x * torch.sigmoid(x)


def get_norm_layer_2d(in_channels, num_groups: int = 32):
    return nn.GroupNorm(num_channels=in_channels, num_groups=num_groups, eps=1e-6, affine=True)


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        temb_channels=512,
        zq_ch=None,
        add_conv=False,
        normalization=get_norm_layer_2d,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(in_channels, zq_channels=zq_ch, add_conv=add_conv)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=(1, 1), padding_mode="replicate"
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = normalization(out_channels, zq_channels=zq_ch, add_conv=add_conv)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=(1, 1), padding_mode="replicate"
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=(1, 1),
                    padding_mode="replicate",
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb, zq=None):
        h = x

        if zq is None:
            h = self.norm1(h)
        else:
            h = self.norm1(h, zq)

        h = nonlinearity(h)  # mb: make in custom act layer.
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        if zq is None:
            h = self.norm2(h)
        else:
            h = self.norm2(h, zq)

        h = nonlinearity(h)
        # h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class PXSDownsample(nn.Module):
    def __init__(self, in_channels: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.spatial_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode="reflect"
        )
        self.linear = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):  # ToDo: add another version with true ps.
        # x: (bchw)
        pxs_interm = self.unshuffle(x)
        b, c, h, w = pxs_interm.shape
        pxs_interm_view = pxs_interm.view(b, c // self.factor**2, self.factor**2, h, w)
        pxs_out = torch.mean(pxs_interm_view, dim=2)

        conv_out = self.spatial_conv(x)

        # adding it all together
        out = conv_out + pxs_out
        return self.linear(out)


class PXSUpsample(nn.Module):
    def __init__(self, in_channels: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"
        )

        self.linear = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        repeated = x.repeat_interleave(self.factor**2, dim=1)
        pxs_interm = self.shuffle(repeated)
        # pxs_out = rearrange(pxs_interm, "(b t) c h w -> b c t h w", t=input_.size(2))

        # Upsampling by 3D-convolution
        image_like_ups = F.interpolate(x, scale_factor=2, mode="nearest")
        # video_like_ups = rearrange(image_like_ups, "(b t) c h w -> b c t h w", t=input_.size(2))
        conv_out = self.spatial_conv(image_like_ups)

        # adding it all together
        out = conv_out + pxs_interm
        return self.linear(out)


def get_upsample_layer(name: str = "native", in_channels: int = 16, out_channels: int = None, factor: int = 2):
    if name == "native":
        return PXSUpsample(in_channels=in_channels, factor=factor)
    else:
        raise ValueError(f"Unrecognized upsample layer name: {name}")


def get_downsample_layer(name: str = "native", in_channels: int = 16, out_channels: int = None, factor: int = 2):
    if name == "native":
        return PXSDownsample(in_channels=in_channels, factor=factor)
    else:
        raise ValueError(f"Unrecognized downsample layer name: {name}")
