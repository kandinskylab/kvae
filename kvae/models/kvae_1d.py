import math
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from ..layers import ResnetBlock1D, Snake1d, WNConv1d, WNConvTranspose1d

# =====================================================
# =================== 1D KVAE Model ===================
# =====================================================


class KVAEAudio(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="KVAE",
    tags=["autoencoder", "audio"],
    repo_url="https://github.com/kandinskylab/kvae",
):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 3, 4, 5, 8],
        latent_dim: int = 2048,
        codebook_dim: Union[int, list] = 64,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 5, 4, 3, 2],
        sample_rate: int = 48000,
        num_channels: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.latent_dim = latent_dim

        if codebook_dim is None:
            codebook_dim = latent_dim
        self.codebook_dim = codebook_dim

        self.hop_length = np.prod(encoder_rates)
        
        self.encoder = Encoder1D(
            d_model=encoder_dim,
            strides=encoder_rates,
            d_latent=latent_dim,
            d_in=num_channels,
        )

        self.in_proj = WNConv1d(latent_dim, codebook_dim * 2, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, latent_dim, kernel_size=1)

        self.decoder = Decoder1D(
            d_latent=latent_dim,
            d_model=decoder_dim,
            strides=decoder_rates,
            d_out=num_channels,
        )

        self.attn = nn.MultiheadAttention(latent_dim, 8, batch_first=True)

    def _pad_to_hop_length(self, audio_data: torch.Tensor) -> torch.Tensor:
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return nn.functional.pad(audio_data, (0, right_pad))

    def encode(self, audio_data: torch.Tensor) -> AutoencoderKLOutput:
        audio_data = self._pad_to_hop_length(audio_data)
        q = self.encoder(audio_data)
        q = q.transpose(1, 2)
        q, _ = self.attn(q, q, q)
        q = q.transpose(1, 2)

        moments = self.in_proj(q)
        posterior = DiagonalGaussianDistribution(moments)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor):
        z = self.out_proj(z)
        z = self.decoder(z)
        return z

    def forward(
        self,
        audio_data: torch.Tensor,
        sample: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        length = audio_data.shape[-1]

        posterior = self.encode(audio_data).latent_dist
        if sample:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "mu": posterior.mean,
            "logvar": posterior.logvar,
        }


# ==================================================
# =================== 1D Encoder ===================
# ==================================================


class EncoderBlock1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResnetBlock1D(input_dim, dilation=1),
            ResnetBlock1D(input_dim, dilation=3),
            ResnetBlock1D(input_dim, dilation=9),
            Snake1d(input_dim),
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder1D(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 3, 4, 5, 8],
        d_latent: int = 64,
        d_in: int = 1,
    ):
        super().__init__()
        layers = [WNConv1d(d_in, d_model, kernel_size=7, padding=3)]

        for stride in strides:
            input_dim = d_model
            d_model *= 2
            layers.append(EncoderBlock1D(input_dim, d_model, stride))

        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.model = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x):
        return self.model(x)


# ==================================================
# =================== 1D Decoder ===================
# ==================================================


class DecoderBlock1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            # odd strides need output_padding=1 to invert the encoder's downsampling exactly, or decode() silently truncates
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResnetBlock1D(output_dim, dilation=1),
            ResnetBlock1D(output_dim, dilation=3),
            ResnetBlock1D(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder1D(nn.Module):
    def __init__(
        self,
        d_latent: int = 64,
        d_model: int = 1536,
        strides: list = [8, 5, 4, 3, 2],
        d_out: int = 1,
    ):
        super().__init__()
        layers = [WNConv1d(d_latent, d_model, kernel_size=7, padding=3)]

        for i, stride in enumerate(strides):
            input_dim = d_model // 2**i
            output_dim = d_model // 2 ** (i + 1)
            layers.append(DecoderBlock1D(input_dim, output_dim, stride))

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)