import torch.nn as nn
import torch.nn.functional as F

from .layers import get_norm_layer_2d


class DecoderSpacialNorm2D(nn.Module):
    def __init__(
        self,
        in_channels,
        zq_channels,
        add_conv=False,
        **norm_layer_params,
    ):
        super().__init__()
        self.norm_layer = get_norm_layer_2d(in_channels, **norm_layer_params)

        self.add_conv = add_conv
        if add_conv:
            self.conv = nn.Conv2d(
                in_channels=zq_channels,
                out_channels=zq_channels,
                kernel_size=3,
                padding=(1, 1),
                padding_mode="replicate",
            )

        self.conv_y = nn.Conv2d(
            in_channels=zq_channels,
            out_channels=in_channels,
            kernel_size=1,
        )
        self.conv_b = nn.Conv2d(
            in_channels=zq_channels,
            out_channels=in_channels,
            kernel_size=1,
        )

    def forward(self, f, zq):
        f_first = f
        f_first_size = f_first.shape[2:]
        zq = F.interpolate(zq, size=f_first_size, mode="nearest")

        if self.add_conv:
            zq = self.conv(zq)

        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f
