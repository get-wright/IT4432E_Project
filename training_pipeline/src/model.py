import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FaceEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)
