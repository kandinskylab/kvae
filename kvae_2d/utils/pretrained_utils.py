"""Utils for loading pretrained models."""

import logging
from typing import List, Union, Iterable, Dict, Any

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _NormBase

logger = logging.getLogger(__name__)


def flatten_modules(modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]]) -> List[nn.Module]:
    """Flattens module into iterable over modules.

    Notes
    -----
    Flattens a module or an iterable of modules into a list of its leaf modules
    (modules with no children) and parent modules that have parameters directly themselves.

    Parameters
    ----------
    modules:
        A given module or an iterable of modules.

    Returns
    -------
    List of modules
    """
    if isinstance(modules, nn.ModuleDict):
        modules = modules.values()

    if isinstance(modules, Iterable):
        _modules = []
        for m in modules:
            _modules.extend(flatten_modules(m))

    else:
        _modules = modules.modules()

    # Capture all leaf modules as well as parent modules that have parameters directly themselves
    return [m for m in _modules if not list(m.children()) or m._parameters]


def make_trainable(modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]]) -> None:
    """Unfreezes the parameters of the provided modules.

    Parameters
    ----------
    modules:
        A given module or an iterable of modules
    """
    modules = flatten_modules(modules)
    for module in modules:
        # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
        for param in module.parameters(recurse=False):
            param.requires_grad = True


def freeze(modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]], train_norm: bool = True) -> None:
    """Freeze module parameters for inference."""
    modules = flatten_modules(modules)
    for mod in modules:
        if isinstance(mod, _NormBase) and train_norm:
            make_trainable(mod)
        else:
            # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
            for param in mod.parameters(recurse=False):
                param.requires_grad = False


def unfreeze(
    modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]],
) -> None:
    """Unfreeze module parameters for training."""
    make_trainable(modules)


def pretrained(
    model: nn.Module,
    ckpt_path: str = None,
    state_dict: Dict[str, Any] = None,
    eval_mode: bool = True,
    freeze_model: bool = True,
    train_norm: bool = False,
    module_name: str = None,
    weights_only: bool = True,
    strict: bool = True,
    device: Union[str, torch.device] = None,
    dtype: torch.dtype = None,
):
    """Loads and freezes pretrained model.

    Parameters
    ----------
    model:
        torch model.
    ckpt_path:
        path to checkpoint.
    eval_mode:
        whether to set the model into eval mode.
    freeze_model:
        whether to freeze the model.
    train_norm:
        Whether to remain norm layers of frozen model trainable.
    module_name:
        if specified will load this name from 'state_dict' of Pytorch Lightning checkpoint.
    weights_only:
        whether to load the weights only from checkpoint.

    Returns
    -------
    loaded model.
    """
    # mb: restore specific loading
    if state_dict is None and ckpt_path is None:
        raise ValueError("ckpt_path or state_dict must be specified.")
    if state_dict is None:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=weights_only)["state_dict"]
    if module_name:
        module_name += "."
        state_dict_to_load = {}
        for key, value in state_dict.items():
            if key.startswith(module_name):
                state_dict_to_load[key.removeprefix(module_name)] = value
        state_dict = state_dict_to_load

    model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded pretrained weights from {ckpt_path}")

    model.train(not eval_mode)
    if freeze_model:
        freeze(model, train_norm=train_norm)
    if dtype is not None:
        model = model.to(dtype)
    if device is not None:
        model = model.to(device)
    return model
