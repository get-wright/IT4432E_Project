"""Load a registry model from a checkpoint and embed an aligned face tensor."""
from __future__ import annotations

from pathlib import Path

import torch

from .embed_wrapper import EmbedWrapper
from .registry import REGISTRY


class Embedder:
    def __init__(self, model_name: str, checkpoint: Path, device: str = "cpu") -> None:
        if model_name not in REGISTRY:
            raise ValueError(f"unknown model {model_name!r}; have {list(REGISTRY)}")
        self.spec = REGISTRY[model_name]
        self.device = device
        self.use_tta = self.spec.use_tta
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        backbone = self.spec.builder(ckpt).to(device)
        self.model = EmbedWrapper(backbone).to(device).eval()
        # Probe embedding dim with a dummy forward at the model's input size.
        s = self.spec.input_size
        with torch.no_grad():
            self.dim = int(self.model.embed_normalized(torch.zeros(1, 3, s, s, device=device)).shape[1])

    @torch.no_grad()
    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        fn = self.model.embed_tta if self.use_tta else self.model.embed_normalized
        return fn(face_tensor).squeeze(0).cpu()
