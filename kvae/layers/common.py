from typing import Union, TypeAlias, Tuple

import torch.nn as nn
import torch

Scalar: TypeAlias = int | float | str | bool | None


def get_activation_with_kwargs(name: str = "swish", **kwargs) -> nn.Module:
    """
    returns an activation function based on the provided name and optional keyword arguments.

    Parameters
    ----------
    name:
        Type of activation function to be used. 
        The default value is 'swish'
    """
    ACT2CLS = {
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }

    if name not in ACT2CLS:
        raise ValueError(f"Activation {name} not supported")

    return ACT2CLS[name](**kwargs)


def cast_tuple(t: Union[Tuple, Scalar], length: int = 1) -> Tuple:
    return t if isinstance(t, tuple) else ((t,) * length)

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)
