"""Generic training utilities: seeding + meters."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Running mean over arbitrary scalar measurements."""

    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0

    def update(self, v: float, n: int = 1) -> None:
        self.sum += v * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / self.n if self.n else 0.0
