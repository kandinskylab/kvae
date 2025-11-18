from typing import Dict

import torch
from torch import nn


class GaussianPrior(nn.Module):
    def __init__(self, mean_weight: float = 1.0):
        """Gaussian prior regularizer.

        Parameters
        ----------
        mean_weight:
            Weight for mean regularization.
        """
        super().__init__()
        self.mean_weight = mean_weight

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean, logvar = torch.chunk(z, 2, dim=1)  # mb: add assert on shape
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        if self.training:
            y_hat = mean + std * torch.randn_like(mean)
        else:
            y_hat = mean
        # we will normalize to number of pixels in the loss
        # this makes training consistent for different video resolutions
        kl_div = 0.5 * torch.mean(self.mean_weight * mean**2 + logvar.exp() - 1 - logvar)
        output_dict = {"y": mean, "y_mean": mean, "y_std": std, "y_hat": y_hat, "kl_div": kl_div}
        return output_dict
