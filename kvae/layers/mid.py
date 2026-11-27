from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn

from .attention import SpatialSelfAttention2D
from .resnet import ResnetBlock2D


class MidBlock2D(nn.Module):
    """Residual blocks with optional attention between adjacent blocks.

    ``norm_factory(channels)`` creates each residual normalization independently.
    Conditional norms bind their own arguments, e.g.
    ``partial(DecoderSpatialNorm2D, zq_ch=32)``, and receive ``zq`` in forward.
    Each residual block reports ``requires_zq`` from its actual normalizations.
    Only those blocks receive ``zq``. Attention keeps its own GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        norm_factory: Callable[[int], nn.Module] = partial(nn.GroupNorm, 32, eps=1e-6, affine=True),
        act_fn: str = "swish",
    ):
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be at least 1")
        self.num_res_blocks = num_res_blocks
        self.use_attention = use_attention
        make_resnet = partial(
            ResnetBlock2D, in_channels=in_channels, norm_factory=norm_factory, act_fn=act_fn,
        )

        self.block_1 = make_resnet()
        for index in range(1, num_res_blocks):
            if use_attention:
                self.add_module(f"attn_{index}", SpatialSelfAttention2D(in_channels))
            self.add_module(f"block_{index + 1}", make_resnet())

    def forward(self, x: torch.Tensor, zq: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.children():
            if isinstance(layer, ResnetBlock2D) and layer.requires_zq:
                x = layer(x, zq=zq)
            else:
                x = layer(x)
        return x
