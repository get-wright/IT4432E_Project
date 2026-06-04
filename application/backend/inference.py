"""Load trained checkpoint and embed a single aligned face tensor.

Supports two checkpoint formats:
- IT4432E training pipeline: keys include "cfg" with training config
- train_local.py IResNet50/AdaFace: keys are "model" and "num_classes"
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


class Embedder:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        self.device = device
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if "cfg" in ckpt:
            from training_pipeline.src.model import FaceEmbedding
            self.dim = ckpt["cfg"]["train"]["embedding_dim"]
            self.model = FaceEmbedding(embedding_dim=self.dim, pretrained=False).to(device)
        else:
            from .iresnet import iresnet50
            self.dim = ckpt["model"]["fc.weight"].shape[0]
            self.model = iresnet50(embedding_size=self.dim).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        emb = self.model(face_tensor)
        return F.normalize(emb, p=2, dim=1).squeeze(0).cpu()
