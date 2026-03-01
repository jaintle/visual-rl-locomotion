"""
Reproducibility seeding utility.

Sets the RNG state for Python's random module, NumPy, and PyTorch
(both CPU and all CUDA devices) from a single integer seed.

Note: Full determinism in MuJoCo/Gymnasium is not guaranteed by seeding
alone, but this is sufficient for stable, comparable experiments.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed all relevant RNG sources.

    Args:
        seed: Non-negative integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
