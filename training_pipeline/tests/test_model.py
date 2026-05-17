import torch
from training_pipeline.src.model import FaceEmbedding


def test_embedding_shape_and_norm():
    m = FaceEmbedding(embedding_dim=512, pretrained=False).eval()
    x = torch.randn(4, 3, 160, 160)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (4, 512)
    norms = y.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_custom_embedding_dim():
    m = FaceEmbedding(embedding_dim=128, pretrained=False).eval()
    x = torch.randn(2, 3, 160, 160)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 128)
