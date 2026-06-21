"""Uniform embedding interface over any backbone returning raw 512-d vectors."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def embed_normalized(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), p=2, dim=1)

    def embed_tta(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.embed_normalized(x)
        e2 = self.embed_normalized(torch.flip(x, dims=[-1]))
        return F.normalize(e1 + e2, p=2, dim=1)
