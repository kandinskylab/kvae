import math
from typing import Tuple, Union, Optional, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import cast_tuple


class SafeConv3d(nn.Conv3d):
    def forward(
        self,
        x: torch.Tensor,
        write_to: Optional[torch.Tensor] = None,
        transform: Optional[Callable] = None,
    ) -> torch.Tensor:
        if transform is None:
            transform = lambda x: x

        memory_count = x.numel() / (10**9)
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = math.ceil(memory_count / 2)
            input_chunks = torch.chunk(x, part_num, dim=2)

            if any(ch.size(2) < kernel_size for ch in input_chunks) and kernel_size > 1:
                assert input_chunks[0].numel() * (
                    kernel_size / input_chunks[0].size(2)
                ) < (2 * 10**9), "frames are too big for Conv3d"

                t_stride, output = self.stride[0], []
                for i in range(0, x.size(2) - kernel_size + 1, t_stride):
                    chunk = transform(x[:, :, i : i + kernel_size])
                    output.append(super(SafeConv3d, self).forward(chunk))
                output = torch.cat(output, dim=2)
                return output

            if write_to is None:
                output = []
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1 :], chunk], dim=2)
                    output.append(super(SafeConv3d, self).forward(transform(z)))
                output = torch.cat(output, dim=2)
                return output
            else:
                time_offset = 0
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1 :], chunk], dim=2)
                    z_time = z.size(2) - (kernel_size - 1)
                    write_to[:, :, time_offset : time_offset + z_time] = super(
                        SafeConv3d, self
                    ).forward(transform(z))
                    time_offset += z_time
                return write_to
        else:
            if write_to is None:
                return super(SafeConv3d, self).forward(transform(x))
            else:
                write_to[...] = super(SafeConv3d, self).forward(transform(x))
                return write_to


# =============================================================================
# Cached layers
# =============================================================================


class CachedCausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding_mode: Literal["zeros", None] = None,
        **kwargs
    ) -> torch.Tensor:
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert (height_kernel_size % 2) and (width_kernel_size % 2)

        self.height_pad = height_kernel_size // 2
        self.width_pad = width_kernel_size // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size
        self.temporal_dim = 2
        self.padding_mode = padding_mode

        self.stride = stride
        self.conv = SafeConv3d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs
        )
        self.cache = None

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        t_stride = self.stride[0]
        padding_3d = (
            self.height_pad,
            self.height_pad,
            self.width_pad,
            self.width_pad,
            0,
            0,
        )
        input_parallel = F.pad(input_, padding_3d, mode="constant" if self.padding_mode == 'zeros' else (self.padding_mode or 'replicate'))

        if self.cache is None:
            first_frame = input_parallel[:, :, :1]
            time_pad_shape = [i for i in first_frame.shape]
            time_pad_shape[2] = self.time_pad
            padding = first_frame.expand(time_pad_shape)
        else:
            padding = self.cache

        out_size = [i for i in input_.shape]
        out_size[1] = self.conv.out_channels
        if t_stride == 2:
            out_size[2] = (input_.size(2) + 1) // 2
        output = torch.empty(tuple(out_size), dtype=input_.dtype, device=input_.device)

        offset_out = math.ceil(
            padding.size(2) / t_stride
        )  # forward on `padding_poisoned` should take exactly this range
        offset_in = offset_out * t_stride - padding.size(
            2
        )  # to make forward on `input_parallel` take slice starting with this index

        if offset_out > 0:
            padding_poisoned = torch.cat(
                [
                    padding,
                    input_parallel[
                        :, :, : offset_in + self.time_kernel_size - t_stride
                    ],
                ],
                dim=2,
            )
            output[:, :, :offset_out] = self.conv(padding_poisoned)

        if offset_out < output.size(2):
            output[:, :, offset_out:] = self.conv(input_parallel[:, :, offset_in:])

        # exact formula, doesn't depend on size of segments
        pad_offset = (
            offset_in
            + t_stride
            * math.trunc(
                (input_parallel.size(2) - offset_in - self.time_kernel_size) / t_stride
            )
            + t_stride
        )

        if (
            pad_offset < 0
        ):  # specific to small chunks (for inference on high resolution videos)
            self.cache = torch.cat([padding[:, :, pad_offset:], input_parallel], dim=2)
        else:
            self.cache = torch.clone(input_parallel[:, :, pad_offset:])

        return output
