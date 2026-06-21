import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FaceEmbedding(nn.Module):
    """ResNet50 backbone + Linear(2048, embedding_dim) head.

    `forward` returns the raw embedding (un-normalized). Use `embed_normalized`
    when you need a unit-norm vector (eval, app inference). The Phase 2 triplet
    loss normalizes internally; the Phase 1 classifier head consumes the raw vector.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, embedding_dim)

    def forward(self, x):
        return self.backbone(x)

    def embed_normalized(self, x):
        return F.normalize(self.forward(x), p=2, dim=1)

    def embed_tta(self, x):
        """L2-normalized average of embeddings from x and its horizontal flip.

        2x forward at inference; helps on profile/pose benchmarks by reducing
        left/right asymmetry. No retraining needed.
        """
        e1 = self.embed_normalized(x)
        e2 = self.embed_normalized(torch.flip(x, dims=[-1]))
        return F.normalize(e1 + e2, p=2, dim=1)


class ClassifierHead(nn.Module):
    """Linear classifier over identity labels for Phase 1 softmax warmup.

    No bias: per-class weight vectors should be comparable in norm — standard
    for face-recognition CE warmup recipes.
    """

    def __init__(self, embedding_dim: int, n_identities: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, n_identities, bias=False)

    def forward(self, emb):
        return self.fc(emb)
