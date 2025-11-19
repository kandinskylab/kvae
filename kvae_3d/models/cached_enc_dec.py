import functools

import numpy as np
import torch.nn as nn

from .cached_layers import CachedPXSDownsample, CachedPXSUpsample
from .cached_layers import CachedCausalConv3d, CachedCausalResnetBlock3D
from .cached_layers import Normalize, Normalize3D
from .layers import RMSNorm
from .layers import nonlinearity


class CachedEncoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        in_channels=3,
        resolution=0,
        z_channels=16,
        double_z=True,
        temporal_compress_times=4,
        gather_norm=False,
        norm_type="group_norm",
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        self.conv_in = CachedCausalConv3d(
            chan_in=in_channels,
            chan_out=self.ch,
            kernel_size=3,
        )

        normalization = Normalize if norm_type == "group_norm" else RMSNorm

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        temb_channels=self.temb_ch,
                        gather_norm=gather_norm,
                        normalization=normalization,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level < self.temporal_compress_level:
                    down.downsample = CachedPXSDownsample(block_in, compress_time=True)
                else:
                    down.downsample = CachedPXSDownsample(block_in, compress_time=False)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            gather_norm=gather_norm,
            normalization=normalization,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            gather_norm=gather_norm,
            normalization=normalization,
        )

        # end
        self.norm_out = normalization(block_in, gather=gather_norm)

        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=2 * z_channels if double_z else z_channels,
            kernel_size=3,
        )

    def forward(self, x, cache_dict, use_cp=True):
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x, cache=cache_dict['conv_in'])
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb, layer_cache=cache_dict[i_level][i_block])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h, cache=cache_dict[i_level]['down'])

        # middle
        h = self.mid.block_1(h, temb, layer_cache=cache_dict['mid_1'])
        h = self.mid.block_2(h, temb, layer_cache=cache_dict['mid_2'])

        # end
        h = self.norm_out(h, cache=cache_dict['norm_out'])
        h = nonlinearity(h)
        h = self.conv_out(h, cache=cache_dict['conv_out'])

        return h
    
    def get_last_layer(self):
        return self.conv_out.conv.weight


class CachedDecoder3D(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=None,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        resamp_with_conv=True,
        resolution=0,
        z_channels=16,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
        pad_mode="first",
        temporal_compress_times=4,
        gather_norm=False,
        norm_type="group_norm",
        **kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        if zq_ch is None:
            zq_ch = z_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = CachedCausalConv3d(
            chan_in=z_channels,
            chan_out=block_in,
            kernel_size=3,
        )

        modulated_norm = functools.partial(Normalize3D, normalization=Normalize if norm_type == "group_norm" else RMSNorm)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=modulated_norm,
            gather_norm=gather_norm,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=modulated_norm,
            gather_norm=gather_norm,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        normalization=modulated_norm,
                        gather_norm=gather_norm,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = CachedPXSUpsample(block_in, compress_time=False)
                else:
                    up.upsample = CachedPXSUpsample(block_in, compress_time=True)
            self.up.insert(0, up)

        self.norm_out = modulated_norm(block_in, zq_ch, add_conv=add_conv) #, gather=gather_norm)

        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=out_ch,
            kernel_size=3,
        )

    def forward(self, z, cache_dict):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        t = z.shape[2]

        zq = z
        h = self.conv_in(z, cache_dict['conv_in'])

        # middle
        h = self.mid.block_1(h, temb, layer_cache=cache_dict['mid_1'], zq=zq)
        h = self.mid.block_2(h, temb, layer_cache=cache_dict['mid_2'], zq=zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, layer_cache=cache_dict[i_level][i_block], zq=zq)

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h, cache_dict[i_level]['up'])

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq, cache_dict['norm_out'])
        h = nonlinearity(h)
        h = self.conv_out(h, cache_dict['conv_out'])

        return h

    def get_last_layer(self):
        return self.conv_out.conv.weight
