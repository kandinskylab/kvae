from typing import Tuple, Optional

import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from huggingface_hub import PyTorchModelHubMixin

from diffusers.models.activations import get_activation
from ..layers.resnet import ResnetBlock2D
from ..layers.sampling import PXSDownsample, PXSUpsample
from ..layers.norm import DecoderSpatialNorm2D


# =====================================================
# =================== 2D KVAE Model ===================
# =====================================================


class KVAE2D(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="KVAE 2D",
    tags=["vae"],
    repo_url="https://github.com/kandinskylab/kvae",
):
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 128,
        num_enc_blocks: int = 2,
        num_dec_blocks: int = 2,
        z_channels: int = 16,
        double_z: bool = True,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.encoder = Encoder2D(
            in_channels=in_channels,
            ch=channels,
            ch_mult=ch_mult,
            num_res_blocks=num_enc_blocks,
            z_channels=z_channels,
            double_z=double_z,
        )
        self.decoder = Decoder2D(
            out_ch=in_channels,
            ch=channels,
            ch_mult=ch_mult,
            num_res_blocks=num_dec_blocks,
            in_channels=None,
            z_channels=z_channels,
        )

    def encode(self, x: torch.Tensor) -> AutoencoderKLOutput:

        latent = self.encoder(x)
        posterior = DiagonalGaussianDistribution(latent)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:

        rec = self.decoder(latent)

        return rec

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:

        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        rec = self.decode(z)

        return rec


# ==================================================
# =================== 2D Encoder ===================
# ==================================================


class Encoder2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        z_channels: int = 16,
        double_z: bool = True,
        act_fn: str = "swish",
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * self.num_resolutions
        else:
            self.num_res_blocks = num_res_blocks

        self.nonlinearity = get_activation(act_fn)

        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.ch,
            kernel_size=3,
            padding=(1, 1),
        )

        in_ch_mult = (ch_mult[0],) + tuple(ch_mult)

        # downsampling
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock2D(
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level < self.num_resolutions - 1:
                down.downsample = PXSDownsample(in_channels=block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
        )

        self.mid.block_2 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
        )

        # end
        self.norm_out = nn.GroupNorm(
            num_channels=block_in, num_groups=32, eps=1e-6, affine=True
        )

        self.conv_out = nn.Conv2d(
            in_channels=block_in,
            out_channels=2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
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
# =================== 2D Decoder ===================
# ==================================================


class Decoder2D(nn.Module):
    def __init__(
        self,
        *,
        out_ch: int = 3,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        in_channels: Optional[int] = None,
        z_channels: int = 16,
        zq_ch: Optional[int] = None,
        act_fn: str = "swish",
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.nonlinearity = get_activation(act_fn)

        if zq_ch is None:
            zq_ch = z_channels

        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = nn.Conv2d(
            in_channels=z_channels,
            out_channels=block_in,
            kernel_size=3,
            padding=(1, 1),
            padding_mode="replicate",
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            zq_ch=zq_ch,
            normalization=DecoderSpatialNorm2D,
        )

        self.mid.block_2 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            zq_ch=zq_ch,
            normalization=DecoderSpatialNorm2D,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock2D(
                        in_channels=block_in,
                        out_channels=block_out,
                        zq_ch=zq_ch,
                        normalization=DecoderSpatialNorm2D,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = PXSUpsample(in_channels=block_in)
            self.up.insert(0, up)

        # end
        self.norm_out = DecoderSpatialNorm2D(block_in, zq_ch)

        self.conv_out = nn.Conv2d(
            in_channels=block_in,
            out_channels=out_ch,
            kernel_size=3,
            padding=(1, 1),
            padding_mode="replicate",
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        h = self.mid.block_1(h, z)
        h = self.mid.block_2(h, z)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, z)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h, z)
        h = self.nonlinearity(h)
        h = self.conv_out(h)

        return h