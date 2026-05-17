"""End-to-end smoke test: a tiny two-phase run on synthetic data must not collapse."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from training_pipeline.src.train import run_training


@pytest.fixture
def synthetic_manifest(tmp_path: Path):
    """Build a tiny manifest with 5 identities x 4 images = 20 samples."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    base_colors = [(220, 20, 60), (50, 205, 50), (30, 144, 255), (255, 215, 0), (148, 0, 211)]
    for i, color in enumerate(base_colors):
        for j in range(4):
            jitter = rng.integers(-20, 20, size=3)
            pixel = tuple(int(np.clip(c + d, 0, 255)) for c, d in zip(color, jitter))
            img = Image.new("RGB", (160, 160), pixel)
            path = img_dir / f"id{i:02d}_{j}.jpg"
            img.save(path)
            rows.append({
                "path": str(path.relative_to(tmp_path)),
                "identity_id": f"id{i:02d}",
                "source": "synthetic",
                "split": "train" if j < 3 else "val",
            })
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest)
    return tmp_path, manifest


class _TinyEmbed(nn.Module):
    def __init__(self, embedding_dim: int = 32, pretrained: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.fc = nn.Linear(16 * 80 * 80, embedding_dim)
        self.backbone = type("X", (), {"fc": self.fc})()

    def forward(self, x):
        h = F.relu(self.conv(x)).flatten(1)
        return self.fc(h)

    def embed_normalized(self, x):
        return F.normalize(self.forward(x), p=2, dim=1)


@pytest.mark.slow
def test_two_phase_smoke_no_collapse(synthetic_manifest, monkeypatch, tmp_path):
    """Run 1 phase-1 epoch + 1 phase-2 epoch on synthetic data. Must not crash; phase-1 gate must pass."""
    workdir, manifest = synthetic_manifest

    import training_pipeline.src.dataset as ds_mod
    monkeypatch.setattr(ds_mod, "ROOT", workdir)

    import training_pipeline.src.train as train_mod

    def _stub_probe(*args, **kwargs):
        return {"mean_acc": 0.5, "spread": 0.10, "pos_sim_mean": 0.6, "neg_sim_mean": 0.5}

    monkeypatch.setattr(train_mod, "_run_lfw_probe", _stub_probe)
    monkeypatch.setattr(train_mod, "load_pairs_txt", lambda p: [])
    monkeypatch.setattr(train_mod, "find_lfw_identity_root", lambda p: tmp_path)

    from training_pipeline.src import model as model_mod
    monkeypatch.setattr(model_mod, "FaceEmbedding", _TinyEmbed)
    monkeypatch.setattr(train_mod, "FaceEmbedding", _TinyEmbed)

    cfg = {
        "manifest": str(manifest),
        "checkpoints_dir": str(tmp_path / "ckpts"),
        "tensorboard_dir": str(tmp_path / "tb"),
        "seed": 42,
        "train": {
            "phase1_epochs": 1,
            "phase2_epochs": 1,
            "batches_per_epoch": 5,
            "phase1_batch_size": 8,
            "p": 4, "k": 2, "margin": 0.3,
            "lr_head": 1e-3, "lr_backbone": 1e-4, "weight_decay": 5e-4,
            "warmup_steps": 2, "num_workers": 0, "embedding_dim": 32,
        },
        "eval": {"pairs_txt": "unused-by-stub", "max_pairs_inloop": 10, "batch_size": 8},
    }

    result = run_training(cfg)
    assert "history" in result
    assert (tmp_path / "ckpts" / "phase1_end.pt").exists()
    assert (tmp_path / "ckpts" / "last.pt").exists()
