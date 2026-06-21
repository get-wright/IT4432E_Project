"""Integration test: alignment + embedding works on a real face image.
Skipped automatically if checkpoint or sample LFW image is missing.
"""
from pathlib import Path

import pytest
import torch

CKPT = Path("application/models/best.pt")
SAMPLE_ROOT = Path("shared/process-data/lfw_pairs")


def _first_sample() -> Path | None:
    if not SAMPLE_ROOT.exists():
        return None
    for p in SAMPLE_ROOT.rglob("*.jpg"):
        return p
    return None


SAMPLE = _first_sample()


@pytest.mark.skipif(
    not CKPT.exists() or SAMPLE is None,
    reason="checkpoint or LFW sample missing — run training first",
)
def test_embedding_shape_and_idempotent():
    from application.backend.face_align import FaceAligner
    from application.backend.inference import Embedder

    aligner = FaceAligner()
    embedder = Embedder(CKPT)
    img_bytes = SAMPLE.read_bytes()
    t = aligner.align(img_bytes)
    assert t is not None
    e1 = embedder.embed(t)
    e2 = embedder.embed(t)
    assert e1.shape == (embedder.dim,)
    assert torch.allclose(e1, e2)
    # L2-normalized embedding.
    assert abs(e1.norm().item() - 1.0) < 1e-4


def test_embedder_use_tta_calls_embed_tta(tmp_path):
    """When constructed with use_tta=True, Embedder must use model.embed_tta."""
    import torch
    import torch.nn.functional as F
    from application.backend.inference import Embedder

    calls = {"embed_normalized": 0, "embed_tta": 0}

    class _Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 160 * 160, 4)
        def embed_normalized(self, x):
            calls["embed_normalized"] += 1
            return F.normalize(self.fc(x.flatten(1)), dim=1)
        def embed_tta(self, x):
            calls["embed_tta"] += 1
            return F.normalize(self.fc(x.flatten(1)), dim=1)

    ckpt = tmp_path / "tiny.pt"
    spy = _Spy()
    torch.save({"model": spy.state_dict(), "cfg": {"train": {"embedding_dim": 4}}}, ckpt)

    # Patch Embedder's model construction to return our spy instead of FaceEmbedding.
    import application.backend.inference as inf
    real_model_cls = inf.FaceEmbedding
    inf.FaceEmbedding = lambda **kw: spy
    try:
        emb = Embedder(ckpt, device="cpu", use_tta=True)
        out = emb.embed(torch.zeros(3, 160, 160))
        assert out.shape == (4,)
        assert calls["embed_tta"] == 1
        assert calls["embed_normalized"] == 0
    finally:
        inf.FaceEmbedding = real_model_cls
