from typing import Tuple, Optional, Literal
import math
import numpy as np
import functools

import torch
import torch.nn as nn
from einops import rearrange
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from huggingface_hub import PyTorchModelHubMixin

from ..layers.common import get_activation_with_kwargs
from ..layers.conv import CachedCausalConv3d
from ..layers.norm import get_normalization
from ..layers.resnet import CachedCausalResnetBlock3D
from ..layers.sampling import (
    CachedPXSDownsample,
    CachedPXSUpsample,
    get_downsample_version_options,
    get_upsample_version_options,
)


# =====================================================
# =================== 3D KVAE Model ===================
# =====================================================


class KVAEVideo(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="KVAE",
    tags=["autoencoder", "video"],
    repo_url="https://github.com/kandinskylab/kvae",
):
    def __init__(self, config):
        super().__init__()
        self.data_config = dict(config.get("data", {}))
        self.config = {
            "encoder": config["model"]["encoder_params"],
            "decoder": config["model"]["decoder_params"],
        }
        self.encoder = CachedEncoder3D(**config["model"]["encoder_params"])
        self.decoder = CachedDecoder3D(**config["model"]["decoder_params"])
        latent_normalization = config["model"].get("latent_normalization")
        self.register_buffer("latent_mean", None)
        self.register_buffer("latent_std", None)
        self.use_latent_norm = False
        if latent_normalization is not None:
            channels = config["model"]["encoder_params"].get("z_channels", 16)
            self.latent_mean = torch.zeros(channels)
            self.latent_std = torch.ones(channels)
            self.use_latent_norm = latent_normalization.get("enabled", False)

    def _normalize_posterior(self, posterior):
        # Transform the existing distribution after its original logvar clamp.
        mean = self.latent_mean.float().view(1, -1, 1, 1, 1)
        std = self.latent_std.float().view(1, -1, 1, 1, 1)
        dtype = posterior.mean.dtype
        posterior.mean = ((posterior.mean.float() - mean) / std).to(dtype)
        posterior.logvar = (posterior.logvar.float() - 2 * std.log()).to(dtype)
        posterior.std = (posterior.std.float() / std).to(dtype)
        posterior.var = (posterior.var.float() / std.square()).to(dtype)
        posterior.parameters = torch.cat([posterior.mean, posterior.logvar], dim=1)
        return posterior

    def _denormalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.float().view(1, -1, 1, 1, 1)
        std = self.latent_std.float().view(1, -1, 1, 1, 1)
        return (z.float() * std + mean).to(z.dtype)

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

    def encode(
        self, 
        x: torch.Tensor, 
        seg_len: int = 16, 
        *, 
        use_latent_norm: Optional[bool] = None
    ) -> AutoencoderKLOutput:
        """Return a Gaussian posterior; None follows the model's normalization default."""
        use_latent_norm = self.use_latent_norm if use_latent_norm is None else use_latent_norm
        if use_latent_norm and self.latent_mean is None:
            raise ValueError("This checkpoint has no latent normalization statistics")
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
        
        # Normalize the posterior using the saved per-channel statistics.
        if use_latent_norm:
            posterior = self._normalize_posterior(posterior)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(
        self, 
        z: torch.Tensor, 
        seg_len: int = 16, 
        *, 
        use_latent_norm: Optional[bool] = None
    ) -> torch.Tensor:
        """Decode latents, undoing normalization when the selected mode is enabled."""
        use_latent_norm = self.use_latent_norm if use_latent_norm is None else use_latent_norm
        if use_latent_norm and self.latent_mean is None:
            raise ValueError("This checkpoint has no latent normalization statistics")
        self._reset_cache(module_name="decoder", arg_names="cache")
        
        # Convert normalized latents back to the decoder's input space.
        if use_latent_norm:
            z = self._denormalize_latents(z)

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
        *,
        use_latent_norm: Optional[bool] = None,
    ) -> torch.Tensor:

        posterior = self.encode(x, seg_len, use_latent_norm=use_latent_norm).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        rec = self.decode(z, seg_len, use_latent_norm=use_latent_norm)

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
        downsample_version: Literal[1, 2, 3] = 1,
        temporal_compress_start_level: int = 0,
        act_fn: str = "swish",
        padding_mode: Literal["zeros", None] = None,
        adaptive_channels: bool = False,
        patchify: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.patchify = patchify
        self.first_down_level = 1 if patchify else 0
        self.nonlinearity = get_activation_with_kwargs(act_fn, inplace=True)

        temporal_compress_level = (
            int(np.log2(temporal_compress_times)) + temporal_compress_start_level
        )

        in_ch_mult = (ch_mult[0],) + tuple(ch_mult)
        normalization = functools.partial(get_normalization, norm_name=norm_type)
        if not patchify:
            self.conv_in = CachedCausalConv3d(
                chan_in=in_channels,
                chan_out=round(in_ch_mult[0] * self.ch),
                kernel_size=3,
                padding_mode=padding_mode,
            )
        else:
            hidden_channels = round(ch * ch_mult[0])
            resblock = functools.partial(
                CachedCausalResnetBlock3D,
                normalization=normalization,
                padding_mode=padding_mode,
            )
            self.conv_patch_1 = CachedCausalConv3d(
                chan_in=in_channels * 4,
                chan_out=hidden_channels,
                kernel_size=1,
                padding_mode=padding_mode,
            )
            self.conv_patch_2 = resblock(in_channels=hidden_channels, out_channels=hidden_channels)
            self.conv_stride = CachedCausalConv3d(
                chan_in=in_channels,
                chan_out=hidden_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding_mode=padding_mode,
            )
            self.conv_post_stride = resblock(in_channels=hidden_channels, out_channels=hidden_channels)
            self.norm_pre_adapter = normalization(hidden_channels)
            self.adapter = resblock(in_channels=hidden_channels, out_channels=round(ch * ch_mult[1]))

        version_options = get_downsample_version_options(downsample_version)
        # Per-downsample doubling flags, also used to set the next level's input channels.
        doubles = []
        for i_level in range(self.num_resolutions - 1):
            double_channels = version_options["double_channels"]
            # Adaptive mode disables doubling when the next level is not wider.
            if adaptive_channels and ch_mult[i_level + 1] <= ch_mult[i_level]:
                double_channels = False
            doubles.append(double_channels)

        # downsampling
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            # The stem replaces level 0 while preserving the original level indices.
            if i_level < self.first_down_level:
                self.down.append(nn.Module())
                continue

            block = nn.ModuleList()
            block_in = round(ch * in_ch_mult[i_level])
            block_out = round(ch * ch_mult[i_level])

            if patchify and i_level == self.first_down_level:
                # The adapter already provides this level's width; down[0] is absent.
                block_in = block_out
            elif i_level > 0 and doubles[i_level - 1]:
                block_in *= 2

            for i_block in range(self.num_res_blocks):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        normalization=normalization,
                        padding_mode=padding_mode,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                compress_time = temporal_compress_start_level <= i_level < temporal_compress_level
                down.downsample = CachedPXSDownsample(
                    block_in, 
                    compress_time=compress_time, 
                    version=downsample_version, 
                    padding_mode=padding_mode,
                    double_channels=doubles[i_level],
                )
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            normalization=normalization,
            padding_mode=padding_mode,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            normalization=normalization,
            padding_mode=padding_mode,
        )

        # end
        self.norm_out = normalization(block_in)

        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding_mode=padding_mode,
        )

    def _input_stem(self, x: torch.Tensor) -> torch.Tensor:
        if self.patchify:
            return self._patchify_stem(x)
        return self.conv_in(x)

    def _patchify_stem(self, x: torch.Tensor) -> torch.Tensor:
        x_patch = rearrange(x, "b c t (h p) (w q) -> b (c p q) t h w", p=2, q=2)
        x_patch = self.conv_patch_1(x_patch)
        x_patch = self.conv_patch_2(x_patch)

        x_stride = self.conv_stride(x)
        x_stride = self.nonlinearity(x_stride)
        x_stride = self.conv_post_stride(x_stride)

        h = self.nonlinearity(x_patch + x_stride)
        h = self.norm_pre_adapter(h)
        return self.adapter(h)

    def forward(self, x: torch.Tensor):
        h = self._input_stem(x)
        for i_level in range(self.first_down_level, self.num_resolutions):
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
        padding_mode: Literal["zeros", None] = None,
        upsample_version: Literal[1, 2] = 1,
        adaptive_channels: bool = False,
    ):
        super().__init__()
        version_options = get_upsample_version_options(upsample_version)
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
            padding_mode=padding_mode,
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
            padding_mode=padding_mode,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            zq_ch=zq_ch,
            normalization=modulated_norm,
            padding_mode=padding_mode,
        )

        # Input multipliers come from the next level because decoder levels run in reverse.
        in_ch_mult = tuple(ch_mult[1:]) + (ch_mult[-1],)
        # halves[i] controls upsample i+1 -> i and the input width of level i.
        halves = []
        for i_level in range(self.num_resolutions - 1):
            halve_channels = version_options["halve_channels"]
            # Adaptive mode disables halving when the next level is not narrower.
            if adaptive_channels and round(ch * ch_mult[i_level]) >= round(ch * ch_mult[i_level + 1]):
                halve_channels = False
            halves.append(halve_channels)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_in = round(ch * in_ch_mult[i_level])
            block_out = round(ch * ch_mult[i_level])

            if i_level < self.num_resolutions - 1 and halves[i_level]:
                block_in //= 2

            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        zq_ch=zq_ch,
                        normalization=modulated_norm,
                        padding_mode=padding_mode,
                    )
                )
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level != 0:
                compress_time = (
                    self.num_resolutions - temporal_compress_start_level
                    > i_level
                    >= self.num_resolutions - self.temporal_compress_level
                )
                up.upsample = CachedPXSUpsample(
                    block_in,
                    compress_time=compress_time,
                    padding_mode=padding_mode,
                    version=upsample_version,
                    halve_channels=halves[i_level - 1],
                )

            self.up.insert(0, up)

        # end
        self.norm_out = modulated_norm(block_in, zq_ch=zq_ch)

        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=out_ch,
            kernel_size=3,
            padding_mode=padding_mode,
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
