import time

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SafeConv3d


class PXSUpsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int=2):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = SafeConv3d(in_channels, in_channels,
                                      kernel_size=(1, 3, 3),
                                      stride=(1, 1, 1),
                                      padding=(0, 1, 1),
                                      padding_mode='reflect')
        
        if self.temporal_compress:
            self.temporal_conv = SafeConv3d(in_channels, in_channels,
                                            kernel_size=(3, 1, 1),
                                            stride=(1, 1, 1),
                                            dilation=(1, 1, 1))

        self.linear = nn.Conv3d(in_channels, in_channels,
                                kernel_size=1,
                                stride=1)

    def spatial_upsample(self, input_):
        def conv_part(x):
            to = torch.empty_like(x)
            out = self.spatial_conv(x, write_to=to)
            return out

        b, t, c, h, w = input_.shape
        input_view = input_.view(b, (t *c), h, w)

        input_interp = F.interpolate(input_view, scale_factor=2, mode='nearest')
        input_interp = input_interp.view(b, t, c, 2 * h, 2 * w)
        input_interp.add_(conv_part(input_interp))
        return input_interp

    def temporal_upsample(self, input_):
        # HERE: only one transformation (via Conv3d)
        time_factor = 1.0 + 1.0 * (input_.size(2) > 1)
        if isinstance(time_factor, torch.Tensor):
            time_factor = time_factor.item()

        # input_ : (T + 1) x H x W
        repeated = input_.repeat_interleave(int(time_factor), dim=2)
        # repeated: (2T + 2) x H x W

        tail = repeated[..., int(time_factor - 1) :, :, :]
        # tail: (2T + 1) x H x W

        padding_3d = (0, 0, 0, 0, 2, 0)
        tail_pad = F.pad(tail, padding_3d, mode="replicate")
        conv_out = self.temporal_conv(tail_pad)
        return conv_out + tail

    def forward(self, x):
        if self.temporal_compress:
            # TEMPORAL UPSAMPLE
            x = self.temporal_upsample(x)

        # SPATIAL UPSAMPLE
        s_out = self.spatial_upsample(x)
        return self.linear(s_out)
