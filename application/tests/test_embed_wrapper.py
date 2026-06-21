import torch
import torch.nn as nn
from application.backend.embed_wrapper import EmbedWrapper


class _FakeBackbone(nn.Module):
    def forward(self, x):  # returns a fixed raw 4-d "embedding" per row
        return torch.tensor([[3.0, 4.0, 0.0, 0.0]]).repeat(x.shape[0], 1)


def test_embed_normalized_is_unit_norm():
    w = EmbedWrapper(_FakeBackbone())
    out = w.embed_normalized(torch.zeros(2, 3, 8, 8))
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_embed_tta_is_unit_norm():
    w = EmbedWrapper(_FakeBackbone())
    out = w.embed_tta(torch.zeros(1, 3, 8, 8))
    assert torch.allclose(out.norm(dim=1), torch.ones(1), atol=1e-5)
