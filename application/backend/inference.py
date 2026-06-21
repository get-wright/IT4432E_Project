"""Load trained checkpoint and embed a single aligned face tensor."""
from __future__ import annotations

from pathlib import Path

import torch

from models.arcface.model import FaceEmbedding


class Embedder:
    def __init__(
        self,
        checkpoint: Path,
        device: str = "cpu",
        use_tta: bool = False,
    ) -> None:
        self.device = device
        self.use_tta = use_tta
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.dim = ckpt["cfg"]["train"]["embedding_dim"]
        self.model = FaceEmbedding(embedding_dim=self.dim, pretrained=False).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        embed_fn = self.model.embed_tta if self.use_tta else self.model.embed_normalized
        return embed_fn(face_tensor).squeeze(0).cpu()
