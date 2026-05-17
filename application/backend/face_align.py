"""MTCNN-based face alignment for the inference path."""
from __future__ import annotations

from io import BytesIO

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class FaceAligner:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.mtcnn = MTCNN(
            image_size=160, margin=0, post_process=False,
            device=device, keep_all=False,
        )
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def align(self, img_bytes: bytes) -> torch.Tensor | None:
        """Return normalized (3,160,160) tensor or None if no face detected."""
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        face = self.mtcnn(img)
        if face is None:
            return None
        # facenet-pytorch returns float in [0,255]; convert via PIL for ImageNet normalize.
        face_np = face.byte().permute(1, 2, 0).cpu().numpy()
        return self.normalize(Image.fromarray(face_np))
