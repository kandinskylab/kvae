from typing import Union, Tuple
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import cast_tuple


class SafeConv3d(nn.Conv3d):
    def forward(self, x, write_to=None, transform=None):
        if transform is None:
            transform = lambda x: x

        memory_count = x.numel() * x.element_size() / (10 ** 9)
        if memory_count > 3:
            kernel_size = self.kernel_size[0]
            part_num = math.ceil(memory_count / 2)
            input_chunks = torch.chunk(x, part_num, dim=2)  # NCTHW

            if input_chunks[0].size(2) < 3 and kernel_size > 1:
                for i in range(x.size(2) - 2):
                    torch.cuda.empty_cache()
                    time.sleep(.2)
                    chunk = transform(x[:, :, i:i+3])
                    write_to[:, :, i:i+1] = super(SafeConv3d, self).forward(chunk)
                return write_to

            if write_to is None:
                output = []
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1:], chunk], dim=2)
                    output.append(super(SafeConv3d, self).forward(transform(z)))
                output = torch.cat(output, dim=2)
                return output
            else:
                time_offset = 0
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1:], chunk], dim=2)
                    z_time = z.size(2) - (kernel_size - 1)
                    write_to[:, :, time_offset:time_offset+z_time] = super(SafeConv3d, self).forward(transform(z))
                    time_offset += z_time
                return write_to
        else:
            if write_to is None:
                return super(SafeConv3d, self).forward(transform(x))
            else:
                write_to[...] = super(SafeConv3d, self).forward(transform(x))
                return write_to


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def identity(module, args):
    if not isinstance(args, tuple):
        return module(args)
    return module(*args)


class CausalConv3d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], stride=(1,1,1), dilation=(1,1,1), **kwargs):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert (height_kernel_size % 2) and (width_kernel_size % 2)

        self.height_pad = height_kernel_size // 2
        self.width_pad = width_kernel_size // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size
        self.stride = stride

        self.conv = SafeConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, input_):

        padding_3d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)
        input_padded = F.pad(input_, padding_3d, mode="replicate")

        output = self.conv(input_padded)
        return output

    
def RMSNorm(in_channels, *args, **kwargs):
    return WanRMS_norm(n_ch=in_channels, bias=False)


class WanRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(self, n_ch: int, bias: bool = False) -> None:
        super().__init__()
        shape = (n_ch, 1, 1, 1)

        self.scale = n_ch ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x, *args, **kwargs):
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias
