from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import SafeConv3d


# ==================================================
# =================== 2D Modules ===================
# ==================================================


class DecoderSpatialNorm2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        zq_ch: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_channels=in_channels, num_groups=32, eps=1e-6, affine=True
        )

        self.conv_y = nn.Conv2d(
            in_channels=zq_ch,
            out_channels=in_channels,
            kernel_size=1,
        )
        self.conv_b = nn.Conv2d(
            in_channels=zq_ch,
            out_channels=in_channels,
            kernel_size=1,
        )

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_first = f
        f_first_size = f_first.shape[2:]
        zq = F.interpolate(zq, size=f_first_size, mode="nearest")

        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


# ==================================================
# =================== 3D Modules ===================
# ==================================================
class RMS_norm(nn.Module):
    def __init__(self, num_channels: int, bias: bool = False):
        super().__init__()
        shape = (num_channels, 1, 1, 1)

        self.scale = num_channels**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class DecoderCachedSpatialNorm3D(nn.Module):
    def __init__(self, in_channels: int, zq_ch: int, normalization: nn.Module):
        super().__init__()
        self.norm_layer = normalization

        self.conv_y = SafeConv3d(
            in_channels=zq_ch,
            out_channels=in_channels,
            kernel_size=1,
        )
        self.conv_b = SafeConv3d(
            in_channels=zq_ch,
            out_channels=in_channels,
            kernel_size=1,
        )
        self.cache = None

    def forward(
        self,
        f: torch.Tensor,
        zq: torch.Tensor,
    ) -> torch.Tensor:

        if self.cache is None:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]

            zq_first = F.interpolate(zq_first, size=f_first_size, mode="nearest")

            if zq.size(2) > 1:
                zq_rest_splits = torch.split(zq_rest, 32, dim=1)
                interpolated_splits = [
                    F.interpolate(split, size=f_rest_size, mode="nearest")
                    for split in zq_rest_splits
                ]

                zq_rest = torch.cat(interpolated_splits, dim=1)
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            f_size = f.shape[-3:]
            zq_splits = torch.split(zq, 32, dim=1)
            interpolated_splits = [
                F.interpolate(split, size=f_size, mode="nearest") for split in zq_splits
            ]
            zq = torch.cat(interpolated_splits, dim=1)

        norm_f = self.norm_layer(f)
        norm_f.mul_(self.conv_y(zq))
        norm_f.add_(self.conv_b(zq))

        self.cache = True

        return norm_f


FIX_NORM_PARAMS = {
    "group_norm": {"num_groups": 32, "eps": 1e-6, "affine": True},
    "rms_norm": {"bias": False},
}

NORMALIZATION_BLOCK = {"group_norm": nn.GroupNorm, "rms_norm": RMS_norm}


def get_normalization(
    in_channels: int,
    norm_name: Literal["group_norm", "rms_norm", "decoder_spatial_norm"] = "group_norm",
    normalization: Optional[Literal["group_norm", "rms_norm"]] = None,
    zq_ch: Optional[int] = None,
) -> nn.Module:
    """
    The function returns a normalization module based on the specified parameters
    and normalization type.

    Parameters
    ----------
    param in_channels:
        number of input channels for the normalization operation
    norm_name:
        specifies the type of normalization to be used
        it can take one of the following values: "group_norm", "rms_norm", or "decoder_spatial_norm", defaults to group_norm
    normalization:
        the type of normalization to be used for DecoderCachedSpatialNorm3D, if we want to use 'decoder_spatial_norm'
    zq_ch:
        number of input channels for the decoder_spatial_norm blocks
    """
    if zq_ch is None and normalization is None:
        return NORMALIZATION_BLOCK[norm_name](
            num_channels=in_channels, **FIX_NORM_PARAMS[norm_name]
        )
    elif (
        norm_name == "decoder_spatial_norm"
        and normalization in NORMALIZATION_BLOCK.keys()
    ):
        return DecoderCachedSpatialNorm3D(
            in_channels,
            zq_ch,
            NORMALIZATION_BLOCK[normalization](
                num_channels=in_channels, **FIX_NORM_PARAMS[normalization]
            ),
        )
    else:
        raise ValueError(
            f"Normalization {norm_name} with {normalization} not supported"
        )
