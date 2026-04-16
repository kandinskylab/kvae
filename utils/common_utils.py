import os
import random
from typing import Tuple

import numpy as np
import torch


def parse_int_tuple(s: str) -> Tuple[int, int]:
    """
    The function takes a string with two comma-separated
    integer values and returns a tuple of those integers.

    Parameters:
    -----------
    s
        input string with two comma-separated integer values

    Returns:
    -------
        A tuple of those integers
    """
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"Exactly 2 comma-separated values are expected. Received: {len(parts)}"
        )
    tuple_vals = tuple(int(p.strip()) for p in parts)
    return tuple_vals


def set_seed_and_optimal_cuda_env(seed: int = 42):
    """
    Captures randomness at all levels
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True