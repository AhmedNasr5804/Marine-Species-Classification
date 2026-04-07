"""
seed.py — Global reproducibility utility.

Call set_seed() once at notebook startup and again before every training run
to guarantee deterministic results across Python, NumPy, and PyTorch.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for full reproducibility.

    Covers:
    - Python built-in random
    - NumPy
    - PyTorch CPU and all CUDA devices
    - cuDNN deterministic mode (disables non-deterministic algorithms)

    Args:
        seed: Integer seed value. Default 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
