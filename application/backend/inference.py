"""Load trained checkpoint and embed a single aligned face tensor."""
from __future__ import annotations

from pathlib import Path

import torch

from training_pipeline.src.model import FaceEmbedding


class Embedder:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        self.device = device
        ckpt = torch.load(checkpoint, map_location=device)
        self.dim = ckpt["cfg"]["train"]["embedding_dim"]
        self.model = FaceEmbedding(embedding_dim=self.dim, pretrained=False).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        return self.model.embed_normalized(face_tensor).squeeze(0).cpu()
