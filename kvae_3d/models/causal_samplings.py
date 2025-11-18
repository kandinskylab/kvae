import time

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SafeConv3d


class PXSDownsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int=2):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.s_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))
        self.spatial_conv = SafeConv3d(in_channels, in_channels,
                                      kernel_size=(1, 3, 3),
                                      stride=(1, 2, 2),
                                      padding=(0, 1, 1),
                                      padding_mode='reflect')
        
        if self.temporal_compress:
            self.temporal_conv = SafeConv3d(in_channels, in_channels,
                                            kernel_size=(3, 1, 1),
                                            stride=(2, 1, 1),
                                            dilation=(1, 1, 1))

        self.linear = nn.Conv3d(in_channels, in_channels,
                                kernel_size=1,
                                stride=1)

    def spatial_downsample(self, input_):

        pxs_out = self.s_pool(input_)

        # Downsampling by 3D-convolution
        conv_out = self.spatial_conv(input_)
        
        # adding it all together
        return conv_out + pxs_out

    def temporal_downsample(self, input_):
        # Interpolation part
        permuted = rearrange(input_, "b c t h w -> (b h w) c t")
        first, rest = permuted[..., :1], permuted[..., 1:]

        if rest.size(-1) > 0:
            rest_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)
            full_interp = torch.cat([first, rest_interp], dim=-1)
        else:
            full_interp = first
        full_interp = rearrange(full_interp, "(b h w) c t -> b c t h w", h=input_.size(-2), w=input_.size(-1))

        # Downsampling by 3D-convolution
        padding_3d = (0, 0, 0, 0, 2, 0)
        input_parallel = F.pad(input_, padding_3d, mode="replicate")
        conv_out = self.temporal_conv(input_parallel)
        
        return conv_out + full_interp

    def forward(self, x):
        # SPATIAL DOWNSAMPLE
        out = self.spatial_downsample(x)
        
        if self.temporal_compress:
            # TEMPORAL DOWNSAMPLE
            out = self.temporal_downsample(out)

        return self.linear(out)


class PXSDownsampleV2(PXSDownsample):
    idx = 0
    def __init__(self, in_channels: int, compress_time: bool, factor: int=2):
        super().__init__(in_channels, compress_time, factor)
        self.layer_idx = PXSDownsampleV2.idx
        PXSDownsampleV2.idx += 1

    def temporal_downsample(self, input_):
        # Interpolation part
        permuted = rearrange(input_, "b c t h w -> (b h w) c t")
        first, rest = permuted[..., :1], permuted[..., 1:]

        if rest.size(-1) > 0:
            rest_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)
        full_interp = torch.cat([first, rest_interp], dim=-1)
        full_interp = rearrange(full_interp, "(b h w) c t -> b c t h w", h=input_.size(-2), w=input_.size(-1))

        # Downsampling by 3D-convolution
        padding_3d = (0, 0, 0, 0, 2, 0)
        input_parallel = F.pad(input_, padding_3d, mode="replicate")

        conv_out = torch.empty_like(full_interp)
        if self.layer_idx == 0:
            conv_out[:, :, :9] = self.temporal_conv(input_parallel[:, :, :19])
            if input_parallel.size(2) > 19:
                conv_out[:, :, 9:] = self.temporal_conv(input_parallel[:, :, 17:])
        else:
            conv_out[:, :, :5] = self.temporal_conv(input_parallel[:, :, :11])
            if input_parallel.size(2) > 11:
                conv_out[:, :, 5:] = self.temporal_conv(input_parallel[:, :, 9:])

        return conv_out + full_interp


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

    def s_ups_1(self, input_rear, inp_time):
        # a = input(f'UPS 1 (0):')
        q = F.interpolate(input_rear, scale_factor=2, mode='nearest')
        # a = input(f'UPS 1 (1):')
        q = rearrange(q, '(b t) c h w -> b c t h w', t=inp_time)
        # a = input(f'UPS 1 (2):')
        to = torch.empty_like(q)
        # a = input(f'UPS 1 (3):')
        q = self.spatial_conv(q, write_to=to)
        # a = input(f'UPS 1 (4):')
        return q

    def s_ups_2(self, input_rear, inp_time):
        z = input_rear.repeat_interleave(self.factor ** 2, dim=1) # МОЖНО ЛИ СЭКОНОМИТЬ НА ЭТОМ МЕТОДЕ?
        z = self.shuffle(z)
        z = rearrange(z, '(b t) c h w -> b c t h w', t=inp_time)
        return z
        
    def spatial_upsample_PREVIOUS(self, input_):
        input_time = input_.size(2)
        input_ = rearrange(input_, 'b c t h w -> (b t) c h w')

        '''
        # PixelShuffle part
        a = input('(0) BEFORE REPEAT:')
        z = input_.repeat_interleave(self.factor ** 2, dim=1)
        a = input('(1) BEFORE SHUFFLE:')
        z = self.shuffle(z)
        a = input('(2) BEFORE REARRANGE:')
        z = rearrange(z, '(b t) c h w -> b c t h w', t=input_time)

        # Upsampling by 3D-convolution
        a = input('(3) BEFORE INTERPOLATE:')
        q = F.interpolate(input_, scale_factor=2, mode='nearest')
        a = input('(4) BEFORE REARRANGE:')
        q = rearrange(q, '(b t) c h w -> b c t h w', t=input_time)
        torch.cuda.empty_cache()
        print(f'Q.SHAPE = {q.shape}')
        a = input('(5) BEFORE CONV:')
        q = self.spatial_conv(q)

        # adding it all together
        # return conv_out + pxs_out
        q.add_(z)
        return q
        '''
        conv_result = self.s_ups_1(input_, inp_time=input_time)
        conv_result.add_(self.s_ups_2(input_, inp_time=input_time))
        return conv_result
        

    def spatial_upsample_NEW(self, input_):
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
        s_out = self.spatial_upsample_NEW(x)
        return self.linear(s_out)


if __name__ == '__main__':
    rand_input = torch.rand((5, 16, 7, 32, 32))
    ds = PXSDownsample(16, 2)
    ups = PXSUpsample(16, 2)

    with torch.no_grad():
        compressed = ds(rand_input)
        output = ups(compressed)

    print(f'INPUT.SHAPE = {rand_input.shape}')
    print(f'COMPRESSED.SHAPE = {compressed.shape}')
    print(f'OUTPUT.SHAPE = {output.shape}')