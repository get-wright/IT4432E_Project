"""MTCNN-based face alignment, parameterized per model (size + normalization)."""
from __future__ import annotations

from io import BytesIO

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms


class FaceAligner:
    def __init__(self, input_size: int, mean, std, device: str = "cpu") -> None:
        self.device = device
        self.input_size = input_size
        self.mtcnn = MTCNN(
            image_size=input_size, margin=0, post_process=False,
            device=device, keep_all=False,
        )
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(list(mean), list(std)),
        ])

    def align(self, img_bytes: bytes) -> torch.Tensor | None:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        face = self.mtcnn(img)
        if face is None:
            return None
        # facenet-pytorch returns float in [0,255]; convert via PIL for normalize.
        face_np = face.byte().permute(1, 2, 0).cpu().numpy()
        return self.normalize(Image.fromarray(face_np))
