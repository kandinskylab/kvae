from typing import Dict

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .decoder import Decoder2D
from .encoder import Encoder2D
from .regularizers import GaussianPrior


class KVAE2D(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="KVAE 2D",
    tags=["vae"],
    repo_url="https://github.com/kandinskylab/kvae-1",
):
    def __init__(
        self,
        in_channels=3,
        channels=128,
        num_enc_blocks=2,
        num_dec_blocks=2,
        z_channels=16,
        double_z=True,
        ch_mult=(1, 2, 4, 8),
        bottleneck: nn.Module = None,
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

        self.bottleneck = bottleneck or GaussianPrior()

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encoder(x)
        output = self.bottleneck(latent)
        return output

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output_dict = self.encode(x)
        output_dict["x_hat"] = self.decode(output_dict["y_hat"])
        return output_dict
