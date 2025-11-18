"""
    Accelerated Kandinsky VAE.
    Dmitry Mikhaylov.

    v1:
        * Fixed image and 17-frame encoding
        * Added compiler cache saving/loading
"""

import math
import os
import time
from math import log2
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cached_enc_dec import CachedDecoder3D


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True

M = 6

RESOLUTIONS = [
    (256, 256), (256, 384), (384, 256),
    (512, 512), (512, 768), (768, 512),
    (1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)
]
FRAMES = [16 * k + 1 for k in range(0, 16)]

class CausalConv3D(nn.Module):
    def __init__(self, c_in, c_out, k_size: Union[int, Tuple[int, int, int]], stride=(1,1,1),
                 dilation=(1,1,1), **kwargs):
        super().__init__()
        k_size = cast_tuple(k_size, 3)
        tks, hks, wks = k_size
        _, hs, ws = stride
        self.h_pad = hks // 2
        self.w_pad = wks // 2
        self.t_pad = tks - 1
        self.padding = (wks//2-ws//2, wks//2-ws//2, hks//2-hs//2, hks//2-hs//2, tks - 1, 0)
        self.conv = nn.Conv3d(c_in, c_out, k_size, stride=stride, dilation=dilation, **kwargs).\
            to(memory_format=torch.channels_last_3d)

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True, 
        "max_autotune": True, "memory_planning": True})
    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.padding, mode="replicate")
        x = self.conv(x)
        return x


class SplittedGN(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, split_size, eps=1e-05, affine=True, device=None,
                 dtype=None):
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)
        self.split_size = split_size
    
    def normalize(self, x: Tensor) -> Tensor:
        func = self.forward
        res = torch.empty_like(x)
        if x.shape[2] in [self.split_size + 1, 1]:
            return func(x)
        else:
            N, C, T, H, W = x.shape
            T -= self.split_size + 1
            rest = x[:, :, (self.split_size + 1):]
            res[:, :, :(self.split_size + 1)] = func(x[:, :, :(self.split_size + 1)])
            N, C, T, H, W = rest.shape
            res[:, :, (self.split_size + 1):] = torch.vmap(func, in_dims=2, out_dims=2)\
                (rest.reshape(N, C, T // self.split_size, self.split_size, H, W))\
                .reshape(N, C, T, H, W)
        return res


class EffDownsample(nn.Module):
    def __init__(self, c_in: int, compress_time: bool, factor: int=2, split_size: int = 16):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.split_size = split_size
        self.s_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))
        self.spatial_conv = nn.Conv3d(c_in, c_in, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                      padding=(0, 1, 1), padding_mode='reflect')
        if compress_time:
            self.t_pool = nn.AvgPool3d((2, 1, 1), (2, 1, 1))
            self.temporal_conv = CausalConv3D(c_in, c_in, k_size=(3, 1, 1), stride=(2, 1, 1))

        self.linear = nn.Conv3d(c_in, c_in, kernel_size=1)

    def spatial(self, x: Tensor) -> Tensor:
        return self.s_pool(x) + self.spatial_conv(x)

    def temporal(self, x: Tensor) -> Tensor:
        N, C, T, H, W = x.shape
        if T == 1:
            return x
        res_pool = torch.empty((N, C, 1 + T // self.factor, H, W), dtype=x.dtype, device=x.device)
        res_pool[:, :, :1] = x[:, :, :1]
        res_pool[:, :, 1:] = self.t_pool(x[:, :, 1:])
        return res_pool + self.temporal_conv(x)


class EffEncoderResnetBlock3D(nn.Module):
    def __init__(self, c_in, c_out=None, split_size=16):
        super().__init__()
        c_out = c_in if c_out is None else c_out
        self.c_in, self.c_out = c_in, c_out
        self.norm1 = SplittedGN(num_groups=32, num_channels=c_in, split_size=split_size, eps=1e-6)
        self.conv1 = CausalConv3D(c_in=c_in, c_out=c_out, k_size=3)
        self.norm2 = SplittedGN(num_groups=32, num_channels=c_out, split_size=split_size, eps=1e-6)
        self.conv2 = CausalConv3D(c_in=c_out, c_out=c_out, k_size=3)
        self.nin_shortcut = nn.Conv3d(c_in, c_out, kernel_size=1) if c_in!=c_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1.normalize(x)
        y = F.silu(y, inplace=True)
        y = self.conv1(y)
        y = self.norm2.normalize(y)
        y = F.silu(y, inplace=True)
        y = self.conv2(y)
        return y + self.nin_shortcut(x)


class KandinskyEncoder3D(nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        in_channels=3,
        z_channels=16,
        double_z=True,
        temporal_compress_times=4,
        **kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        self.conv_in = CausalConv3D(c_in=4, c_out=self.ch, k_size=3)

        split_size = 16
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(EffEncoderResnetBlock3D(c_in=block_in, c_out=block_out,
                                                     split_size=split_size))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                if i_level < self.temporal_compress_level:
                    down.downsample = EffDownsample(block_in, compress_time=True,
                                                    split_size=split_size)
                    split_size //= 2
                else:
                    down.downsample = EffDownsample(block_in, compress_time=False,
                                                    split_size=split_size)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = EffEncoderResnetBlock3D(c_in=block_in, c_out=block_in,
                                                   split_size=split_size)
        self.mid.block_2 = EffEncoderResnetBlock3D(c_in=block_in, c_out=block_in,
                                                   split_size=split_size)

        # end
        self.norm_out = SplittedGN(num_groups=32, num_channels=block_in, split_size=split_size,
                                   eps=1e-6)
        self.conv_out = CausalConv3D(
            c_in=block_in,
            c_out=2 * z_channels if double_z else z_channels,
            k_size=3,
        )
        self.z_channels = z_channels

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True,
                    "max_autotune": True, "memory_planning": True,
                    'max_autotune_conv_backends': 'ATEN,TRITON,CUTLASS'})
    def piece_0(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        x = self.down[0].block[0](x)              # M
        x = self.down[0].block[1](x)              # M
        x = self.down[0].downsample.spatial(x)    # M / 4
        return x # 128, T, H//2, W//2
    
    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True,
                    "max_autotune": True, "memory_planning": True,
                    'max_autotune_conv_backends': 'ATEN,TRITON,CUTLASS'})
    def piece_1(self, x: Tensor) -> Tensor:
        x = self.down[0].downsample.temporal(x)   # M / 8
        x = self.down[0].downsample.linear(x)     # M / 8
        x = self.down[1].block[0](x)              # M / 4
        x = self.down[1].block[1](x)              # M / 4
        x = self.down[1].downsample.spatial(x)    # M / 16
        return x  # 256, (T-1)//2+1, H//4, W//4
    
    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True,
                    "max_autotune": True, "memory_planning": True,
                    'max_autotune_conv_backends': 'ATEN,TRITON,CUTLASS'})
    def piece_2(self, x: Tensor) -> Tensor:
        x = self.down[1].downsample.temporal(x)   # M / 32
        x = self.down[1].downsample.linear(x)     # M / 32
        x = self.down[2].block[0](x)              # M / 16
        x = self.down[2].block[1](x)              # M / 16
        x = self.down[2].downsample.spatial(x)    # M / 64
        x = self.down[2].downsample.linear(x)     # M / 64
        x = self.down[3].block[0](x)              # M / 32
        x = self.down[3].block[1](x)              # M / 32
        x = self.mid.block_1(x)                   # M / 32
        x = self.mid.block_2(x)                   # M / 32
        x = self.norm_out.normalize(x)
        x = F.silu(x, inplace=True)
        x = self.conv_out(x)[:, :self.z_channels]
        return x.contiguous()

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True, 
        "max_autotune": True, "memory_planning": True})
    def forward_1(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.piece_0(x)
        x = self.piece_1(x)
        x = self.piece_2(x)
        return x

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True, 
        "max_autotune": True, "memory_planning": True})
    def forward_2h(self, x: Tensor, th: int, tw: int) -> Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 128, T, H//2, W//2), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(2):
            h2l, h2r = H//2*i, H//2*i+H//2
            h4l, h4r = H//4*i, H//4*i+H//4
            z[0,:,:,h4l:h4r,:] = self.piece_0(x[[0],:,:,h2l:h2r,:])
        x = self.piece_1(z)
        x = self.piece_2(x)
        return x

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True, 
        "max_autotune": True, "memory_planning": True})
    def forward_2w(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 128, T, H//2, W//2), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for j in range(2):
            w2l, w2r = W//2*j, W//2*j+W//2
            w4l, w4r = W//4*j, W//4*j+W//4
            z[0,:,:,:,w4l:w4r] = self.piece_0(x[[0],:,:,:,w2l:w2r])
        x = self.piece_1(z)
        x = self.piece_2(x)
        return x

    def forward_4(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 128, T, H//2, W//2), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(2):
            for j in range(2):
                h2l, h2r, w2l, w2r = H//2*i, H//2*i+H//2, W//2*j, W//2*j+W//2
                h4l, h4r, w4l, w4r = H//4*i, H//4*i+H//4, W//4*j, W//4*j+W//4
                z[0,:,:,h4l:h4r,w4l:w4r] = self.piece_0(x[[0],:,:,h2l:h2r,w2l:w2r])
        x = self.piece_1(z)
        x = self.piece_2(x)
        return x

    def forward_8h(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 256, (T-1)//2+1, H//4, W//4), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(4):
            for j in range(2):
                h4l, h4r, w2l, w2r = H//4*i, H//4*i+H//4, W//2*j, W//2*j+W//2
                h16l, h16r, w8l, w8r = H//16*i, H//16*i+H//16, W//8*j, W//8*j+W//8
                y = self.piece_0(x[[0],:,:,h4l:h4r,w2l:w2r])
                z[0,:,:,h16l:h16r,w8l:w8r] = self.piece_1(y)
        x = self.piece_2(z)
        return x
    
    def forward_8w(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 256, (T-1)//2+1, H//4, W//4), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(2):
            for j in range(4):
                h2l, h2r, w4l, w4r = H//2*i, H//2*i+H//2, W//4*j, W//4*j+W//4
                h8l, h8r, w16l, w16r = H//8*i, H//8*i+H//8, W//16*j, W//16*j+W//16
                y = self.piece_0(x[[0],:,:,h2l:h2r,w4l:w4r])
                z[0,:,:,h8l:h8r,w16l:w16r] = self.piece_1(y)
        x = self.piece_2(z)
        return x

    def forward_16(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 256, (T-1)//2+1, H//4, W//4), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(4):
            for j in range(4):
                h4l, h4r, w4l, w4r = H//4*i, H//4*i+H//4, W//4*j, W//4*j+W//4
                h16l, h16r, w16l, w16r = H//16*i, H//16*i+H//16, W//16*j, W//16*j+W//16
                y = self.piece_0(x[[0],:,:,h4l:h4r,w4l:w4r])
                z[0,:,:,h16l:h16r,w16l:w16r] = self.piece_1(y)
        x = self.piece_2(z)
        return x

class SafeConv3d(nn.Conv3d):
    def forward(self, x, write_to=None, transform=None):
        if transform is None:
            transform = lambda x: x

        memory_count = x.numel() * x.element_size() / (10 ** 9)
        if memory_count > 3:
            kernel_size = self.kernel_size[0]
            part_num = math.ceil(memory_count / 2)
            input_chunks = torch.chunk(x, part_num, dim=2)  # NCTHW
            # if kernel_size > 1:
            #     input_chunks = [input_chunks[0]] + [
            #         torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)
            #         for i in range(1, len(input_chunks))
            #     ]
            # else:
            #     input_chunks = [torch.clone(chunk) for chunk in input_chunks]

            if input_chunks[0].size(2) < 3 and kernel_size > 1:
                for i in range(x.size(2) - 2):
                    torch.cuda.empty_cache()
                    time.sleep(.2)
                    # a = input(f'DUMMY ENTER ({i} / {x.size(2) - 2}):')
                    chunk = transform(x[:, :, i:i+3])
                    write_to[:, :, i:i+1] = super(SafeConv3d, self).forward(chunk)
                    # a = input(f'DUMMY ENTER ({i + 1} / {x.size(2) - 2}):')
                # output = torch.cat(output, dim=2)
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


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


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
        self.temporal_dim = 2

        self.stride = stride
        self.conv = SafeConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        self.cache_padding = None

    def forward(self, input_):
        input_parallel = input_

        padding_3d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)
        input_parallel = F.pad(input_parallel, padding_3d, mode="replicate")

        output = self.conv(input_parallel)
        return output


class KandinskyVAE(nn.Module):
    def __init__(self, encoder_conf, decoder_conf, ckpt_path=None):
        super().__init__()
        self.encoder = KandinskyEncoder3D(**encoder_conf)
        self.decoder = CachedDecoder3D(**decoder_conf)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        
        self.encoder = self.encoder.to(memory_format=torch.channels_last_3d)
        self.model_size = sum(p.numel() for p in self.parameters()) * 2

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["state_dict"]

        # Fix checkpoint if starting from new style
        replace_keys = dict()
        delete_keys = []
        for k in sd:
            if "loss" in k:
                delete_keys.append(k)
            
            if "encoder.conv_in.conv.weight" in k:
                sd[k] = F.pad(sd[k], (0, 0, 0, 0, 0, 0, 0, 1))

            if 'encoder.down' in k and 'downsample.temporal_conv.conv' in k:
                continue

            if k.startswith('decoder'):
                if k.startswith('decoder.up.2.upsample') or k.startswith('decoder.up.3.upsample'):
                    continue
                elif '.conv_b.conv' in k:
                    replace_keys[k] = k.replace('.conv_b.conv', '.conv_b')
                    continue
                elif '.conv_y.conv' in k:
                    replace_keys[k] = k.replace('.conv_y.conv', '.conv_y')
                    continue
                
            if k.endswith('sample.temporal_conv.conv.weight') or k.endswith('sample.temporal_conv.conv.bias'):
                replace_keys[k] = k.replace(".temporal_conv.conv.", '.temporal_conv.')
        for k in delete_keys:
            del sd[k]
            if k in replace_keys:
                del replace_keys[k]
        for old_k, new_k in replace_keys.items():
            sd[new_k] = sd[old_k]
            del sd[old_k]

        self.load_state_dict(sd, strict=True)
        print(f'Restored weights from {path}')

    def make_empty_cache(self, block: str):
        def make_dict(name):
            if name == 'conv':
                return {'padding' : None}

            layer, module = name.split('_')
            if layer == 'norm':
                if module == 'enc':
                    return {'mean' : None,
                            'var' : None}
                else:
                    return {'norm' : make_dict('norm_enc'),
                            'add_conv' : make_dict('conv')}
            elif layer == 'resblock':
                return {'norm1' : make_dict(f'norm_{module}'),
                        'norm2' : make_dict(f'norm_{module}'),
                        'conv1' : make_dict('conv'),
                        'conv2' : make_dict('conv'),
                        'conv_shortcut' : make_dict('conv')}
            elif layer.isdigit():
                return {0 : make_dict(f'resblock_{module}'),
                        1 : make_dict(f'resblock_{module}'),
                        2 : make_dict(f'resblock_{module}'),
                        'down' : make_dict('conv'),
                        'up' : make_dict('conv')}

        cache = {'conv_in' : make_dict('conv'),
                 'mid_1' : make_dict(f'resblock_{block}'),
                 'mid_2' : make_dict(f'resblock_{block}'),
                 'norm_out' : make_dict(f'norm_{block}'),
                 'conv_out' : make_dict('conv'),
                 0 : make_dict(f'0_{block}'),
                 1 : make_dict(f'1_{block}'),
                 2 : make_dict(f'2_{block}'),
                 3 : make_dict(f'3_{block}')}

        return cache

    def encode(self, x: Tensor) -> Tensor:
        N, _, T, H, W = x.shape
        if N > 1:
            raise Exception("Batch size greater than 1 does not supported.")
        n_tiles_log = round(log2(128 * T * H * W / 2**31))
        #gpu_mem = torch.cuda.get_device_properties(0).total_memory

        with torch._dynamo.utils.disable_cache_limit():
            if n_tiles_log <= 0:
                return self.encoder.forward_1(x)
            if n_tiles_log <= 1:
                if H > W:
                    return self.encoder.forward_2h(x)
                else:
                    return self.encoder.forward_2w(x)
            if n_tiles_log <= 2:
                return self.encoder.forward_4(x)
            if n_tiles_log <= 3:
                if H > W:
                    return self.encoder.forward_8h(x)
                else:
                    return self.encoder.forward_8w(x)
            else:
                return self.encoder.forward_16(x)

    def decode(self, z: Tensor, seg_len=16) -> Tensor:
        cache = self.make_empty_cache('dec')

        ## get segments size
        split_list = [seg_len + 1]
        n_frames = 4 * (z.size(2) - 1) - seg_len
        while n_frames > 0:
            split_list.append(seg_len)
            n_frames -= seg_len
        split_list[-1] += n_frames
        split_list = [math.ceil(size / 4) for size in split_list]

        ## decode by segments
        recs = []
        for chunk in torch.split(z, split_list, dim=2):
            out = self.decoder(chunk, cache)
            recs.append(out)

        recs = torch.cat(recs, dim=2)
        return recs

    def forward(self, x: Tensor) -> Tensor:
        latent = self.encode(x)
        recs = self.decode(latent)
        return recs
