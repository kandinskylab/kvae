from typing import Tuple, Optional, Literal
import math
import numpy as np
import functools

import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from huggingface_hub import PyTorchModelHubMixin

from ..layers.common import get_activation_with_kwargs
from ..layers.conv import CachedCausalConv3d
from ..layers.norm import get_normalization
from ..layers.resnet import CachedCausalResnetBlock3D
from ..layers.sampling import CachedPXSDownsample, CachedPXSUpsample


# =====================================================
# =================== 3D KVAE Model ===================
# =====================================================


class KVAE3D(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="KVAE 3D",
    tags=["vae"],
    repo_url="https://github.com/kandinskylab/kvae",
):
    def __init__(self, config):
        super().__init__()
        self.config = {
            "encoder": config["model"]["encoder_params"],
            "decoder": config["model"]["decoder_params"],
        }
        self.encoder = CachedEncoder3D(**config["model"]["encoder_params"])
        self.decoder = CachedDecoder3D(**config["model"]["decoder_params"])

    def _reset_cache(self, module_name: str = None, arg_names: str = "cache"):
        def _reset(m):
            if hasattr(m, arg_names):
                setattr(m, arg_names, None)

        if module_name is None:
            module = self
        else:
            module = getattr(self, module_name)

        module.apply(_reset)

    @staticmethod
    def _build_split_list(t_len: int, seg_len: int = 16) -> list[int]:
        split_list = [seg_len + 1]
        n_frames = t_len - (seg_len + 1)
        while n_frames > 0:
            split_list.append(seg_len)
            n_frames -= seg_len
        split_list[-1] += n_frames
        return split_list

    def encode(self, x: torch.Tensor, seg_len: int = 16) -> AutoencoderKLOutput:
        self._reset_cache(module_name="encoder", arg_names="cache")

        # get segments size
        split_list = self._build_split_list(x.size(2), seg_len)

        # encode by segments
        latents = []
        for chunk in torch.split(x, split_list, dim=2):
            l = self.encoder(chunk)
            latents.append(l)
        self._reset_cache(module_name="encoder", arg_names="cache")

        latent = torch.cat(latents, dim=2)
        posterior = DiagonalGaussianDistribution(latent)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, seg_len: int = 16) -> torch.Tensor:
        self._reset_cache(module_name="decoder", arg_names="cache")

        temporal_compress = self.config["decoder"]["temporal_compress_times"]

        # get segments size
        t_len = temporal_compress * (z.size(2) - 1) + 1
        split_list = self._build_split_list(t_len, seg_len)
        split_list = [math.ceil(size / temporal_compress) for size in split_list]

        # decode by segments
        recs = []
        for chunk in torch.split(z, split_list, dim=2):
            out = self.decoder(chunk)
            recs.append(out)

        self._reset_cache(module_name="decoder", arg_names="cache")

        recs = torch.cat(recs, dim=2)
        return recs

    def forward(
        self,
        x: torch.Tensor,
        seg_len: int = 16,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:

        posterior = self.encode(x, seg_len).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        rec = self.decode(z, seg_len)

        return rec


# ==================================================
# =================== 3D Encoder ===================
# ==================================================


class CachedEncoder3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        z_channels: int = 16,
        double_z: bool = True,
        temporal_compress_times: int = 4,
        norm_type: Literal["group_norm", "rms_norm"] = "group_norm",
        downsample_version: Literal[1, 2] = 1,
        temporal_compress_start_level: int = 0,
        act_fn: str = "swish",
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.nonlinearity = get_activation_with_kwargs(act_fn, inplace=True)

        temporal_compress_level = (
            int(np.log2(temporal_compress_times)) + temporal_compress_start_level
        )

        in_ch_mult = (ch_mult[0],) + tuple(ch_mult)
        self.conv_in = CachedCausalConv3d(
            chan_in=in_channels,
            chan_out=round(in_ch_mult[0] * self.ch),
            kernel_size=3,
        )

        normalization = functools.partial(get_normalization, norm_name=norm_type)

        # downsampling
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = round(ch * in_ch_mult[i_level])
            block_out = round(ch * ch_mult[i_level])

            if downsample_version > 1 and i_level > 0:
                block_in *= 2

            for i_block in range(self.num_res_blocks):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        normalization=normalization,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                if temporal_compress_start_level <= i_level < temporal_compress_level:
                    down.downsample = CachedPXSDownsample(
                        block_in, compress_time=True, version=downsample_version
                    )
                else:
                    down.downsample = CachedPXSDownsample(
                        block_in, compress_time=False, version=downsample_version
                    )
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            normalization=normalization,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            normalization=normalization,
        )

        # end
        self.norm_out = normalization(block_in)

        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=2 * z_channels if double_z else z_channels,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)

        return h


# ==================================================
# =================== 3D Decoder ===================
# ==================================================


class CachedDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        out_ch: int = 3,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        z_channels: int = 16,
        temporal_compress_times: int = 4,
        norm_type: Literal["group_norm", "rms_norm"] = "group_norm",
        temporal_compress_start_level: int = 0,
        act_fn: str = "swish",
        zq_ch: Optional[int] = None,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.nonlinearity = get_activation_with_kwargs(act_fn, inplace=True)

        self.temporal_compress_level = (
            int(np.log2(temporal_compress_times)) + temporal_compress_start_level
        )

        if zq_ch is None:
            zq_ch = z_channels

        block_in = round(ch * ch_mult[self.num_resolutions - 1])

        self.conv_in = CachedCausalConv3d(
            chan_in=z_channels,
            chan_out=block_in,
            kernel_size=3,
        )

        modulated_norm = functools.partial(
            get_normalization, norm_name="decoder_spatial_norm", normalization=norm_type
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            zq_ch=zq_ch,
            normalization=modulated_norm,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            zq_ch=zq_ch,
            normalization=modulated_norm,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = round(ch * ch_mult[i_level])
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        zq_ch=zq_ch,
                        normalization=modulated_norm,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                if (
                    self.num_resolutions - temporal_compress_start_level
                    > i_level
                    >= self.num_resolutions - self.temporal_compress_level
                ):
                    up.upsample = CachedPXSUpsample(block_in, compress_time=True)
                else:
                    up.upsample = CachedPXSUpsample(block_in, compress_time=False)
            self.up.insert(0, up)

        # end
        self.norm_out = modulated_norm(block_in, zq_ch=zq_ch)

        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=out_ch,
            kernel_size=3,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        h = self.mid.block_1(h, zq=z)
        h = self.mid.block_2(h, zq=z)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, zq=z)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h, z)
        h = self.nonlinearity(h)
        h = self.conv_out(h)

        return h