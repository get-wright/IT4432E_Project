"""Single source of truth for per-model inference contracts + backbone builders.

Each builder takes a loaded checkpoint dict and returns a backbone whose
forward(x) is the raw (un-normalized) 512-d embedding, weights loaded strict=True.
Normalization to unit length is added later by EmbedWrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt_filename: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    ckpt_key: str          # key under which the state_dict lives in the checkpoint
    needs_cfg: bool         # arcface stores embedding_dim under ckpt["cfg"]
    threshold: float
    builder: Callable[[dict], nn.Module]
    use_tta: bool = False


def build_arcface(ckpt: dict) -> nn.Module:
    from models.arcface.model import FaceEmbedding
    dim = ckpt["cfg"]["train"]["embedding_dim"]
    model = FaceEmbedding(embedding_dim=dim, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.eval()


def build_adaface(ckpt: dict) -> nn.Module:
    from models.adaface.iresnet import iresnet50
    sd = ckpt["model"]
    dim = sd["fc.weight"].shape[0]
    model = iresnet50(embedding_size=dim)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def build_facenet(ckpt: dict) -> nn.Module:
    # Mirror the notebook: backbone held as .bb so bb.-prefixed keys load directly.
    from facenet_pytorch import InceptionResnetV1

    class FaceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bb = InceptionResnetV1(pretrained=None, classify=False)

        def forward(self, x):
            return self.bb(x)  # raw embedding; EmbedWrapper normalizes

    model = FaceNet()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval()


REGISTRY: dict[str, ModelSpec] = {
    "arcface": ModelSpec("arcface", "arcface.pt", 160, *IMAGENET,
                         ckpt_key="model", needs_cfg=True, threshold=0.565,
                         builder=build_arcface, use_tta=False),
    "adaface": ModelSpec("adaface", "adaface.pt", 112, *HALF,
                         ckpt_key="model", needs_cfg=False, threshold=0.5,
                         builder=build_adaface),
    "facenet": ModelSpec("facenet", "facenet.pth", 160, *HALF,
                         ckpt_key="model_state_dict", needs_cfg=False, threshold=0.5,
                         builder=build_facenet),
}
