"""Integration test: alignment + embedding works on a real face image.
Skipped automatically if checkpoint or sample LFW image is missing.
"""
from pathlib import Path

import pytest
import torch

CKPT = Path("application/models/best.pt")
SAMPLE_ROOT = Path("process-data/lfw_pairs")


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
