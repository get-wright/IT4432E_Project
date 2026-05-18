"""ArcFace angular-margin head (standard non-easy-margin variant from the paper).

Consumes L2-normalized embeddings, returns scaled logits ready for cross-entropy.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = float(s)
        self.m = float(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # Wrong-hemisphere boundary: cos(pi - m). Below this, fall back to cos_theta - mm.
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, D] — already L2-normalized by caller.
        weight_norm = F.normalize(self.weight, dim=1)
        cos_theta = embeddings @ weight_norm.t()
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Standard fallback for wrong-hemisphere (NOT easy-margin): cos_theta - mm.
        cos_theta_m = torch.where(
            cos_theta > self.threshold,
            cos_theta_m,
            cos_theta - self.mm,
        )

        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(cos_theta.dtype)
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        return logits * self.s

    @torch.no_grad()
    def center_norms(self) -> torch.Tensor:
        """Per-class L2 norms of the weight matrix — used to diagnose center collapse."""
        return self.weight.norm(dim=1)
