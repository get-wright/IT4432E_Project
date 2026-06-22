"""Integration test: alignment + embedding works on a real face image.
Skipped automatically if checkpoint or sample LFW image is missing.
"""
from pathlib import Path

import pytest
import torch
from application.backend.registry import REGISTRY

CKPT = Path("application/models/arcface.pt")
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

    spec = REGISTRY["arcface"]
    aligner = FaceAligner(spec.input_size, spec.mean, spec.std)
    embedder = Embedder("arcface", CKPT)
    img_bytes = SAMPLE.read_bytes()
    t = aligner.align(img_bytes)
    assert t is not None
    e1 = embedder.embed(t)
    e2 = embedder.embed(t)
    assert e1.shape == (embedder.dim,)
    assert torch.allclose(e1, e2)
    # L2-normalized embedding.
    assert abs(e1.norm().item() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Registry-dispatch unit test (no real weights)
# ---------------------------------------------------------------------------
from application.backend import inference as inf


def test_embedder_uses_tta_flag():
    # Build a fake wrapper recording which path was called.
    calls = {"normal": 0, "tta": 0}

    class FakeWrapper:
        def embed_normalized(self, x):
            calls["normal"] += 1
            return torch.zeros(x.shape[0], 4)

        def embed_tta(self, x):
            calls["tta"] += 1
            return torch.zeros(x.shape[0], 4)

    e = inf.Embedder.__new__(inf.Embedder)   # bypass __init__/checkpoint load
    e.device = "cpu"
    e.model = FakeWrapper()
    e.use_tta = True
    e.embed(torch.zeros(3, 8, 8))
    assert calls == {"normal": 0, "tta": 1}
