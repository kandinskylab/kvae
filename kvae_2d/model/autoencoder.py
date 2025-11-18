from typing import Tuple, Optional, Union, Dict

import torch
import torch.nn as nn


class DefaultBottleneck(nn.Module):
    def __init__(self):
        super(DefaultBottleneck, self).__init__()

    def forward(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"y": y, "y_hat": y}

    def compress(self, y: torch.Tensor) -> torch.Tensor:
        return y


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        bottleneck: Optional[nn.Module] = None,
        latent_shape: Union[Tuple[int, ...], int] = None,
    ):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck or DefaultBottleneck()
        self.latent_shape = latent_shape

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
