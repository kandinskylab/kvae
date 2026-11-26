import torch
import torch.nn as nn


class SpatialSelfAttention2D(nn.Module):
    """Single-head spatial attention used by the image middle block."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        batch, channels, height, width = q.shape
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        weights = torch.einsum("bij,bjk->bik", q, k) * channels**-0.5
        weights = weights.softmax(dim=2)
        h = torch.einsum("bij,bjk->bik", v.flatten(2), weights.transpose(1, 2))
        h = h.reshape(batch, channels, height, width)
        return x + self.proj_out(h)
