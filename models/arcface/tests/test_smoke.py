"""End-to-end smoke test: a tiny ArcFace two-phase run on synthetic data."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


@pytest.fixture
def synthetic_manifest(tmp_path: Path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    base_colors = [
        (220, 20, 60), (50, 205, 50), (30, 144, 255), (255, 215, 0),
        (148, 0, 211), (255, 99, 71), (60, 179, 113), (123, 104, 238),
    ]
    for i, color in enumerate(base_colors):
        for j in range(8):
            jitter = rng.integers(-15, 15, size=3)
            pixel = tuple(int(np.clip(c + d, 0, 255)) for c, d in zip(color, jitter))
            img = Image.new("RGB", (160, 160), pixel)
            path = img_dir / f"id{i:02d}_{j}.jpg"
            img.save(path)
            split = "train" if j < 7 else "val"
            rows.append({
                "path": str(path.relative_to(tmp_path)),
                "identity_id": f"id{i:02d}",
                "source": "synthetic",
                "split": split,
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

    def embed_tta(self, x):
        return F.normalize(self.embed_normalized(x) + self.embed_normalized(torch.flip(x, dims=[-1])), dim=1)


@pytest.mark.slow
def test_arcface_smoke_no_collapse(synthetic_manifest, monkeypatch, tmp_path):
    workdir, manifest = synthetic_manifest

    import models.arcface.dataset as ds_mod
    monkeypatch.setattr(ds_mod, "ROOT", workdir)

    import models.arcface.train as train_mod

    def _stub_probe(*args, **kwargs):
        return {"mean_acc": 0.55, "spread": 0.12, "pos_sim_mean": 0.6, "neg_sim_mean": 0.48}

    monkeypatch.setattr(train_mod, "_run_lfw_probe", _stub_probe)
    monkeypatch.setattr(train_mod, "load_pairs_txt", lambda p: [])
    monkeypatch.setattr(train_mod, "find_lfw_identity_root", lambda p: tmp_path)

    from models.arcface import model as model_mod
    monkeypatch.setattr(model_mod, "FaceEmbedding", _TinyEmbed)
    monkeypatch.setattr(train_mod, "FaceEmbedding", _TinyEmbed)

    cfg = {
        "manifest": str(manifest),
        "checkpoints_dir": str(tmp_path / "ckpts"),
        "tensorboard_dir": str(tmp_path / "tb"),
        "seed": 42,
        "checkpoint_name": "best_arcface.pt",
        "train": {
            "phase1_epochs": 1,
            "phase2_epochs": 3,
            "batches_per_epoch": 5,
            "phase1_batch_size": 8,
            "p": 4, "k": 2,
            "lr_head": 1e-3, "lr_backbone": 1e-4, "weight_decay": 5e-4,
            "warmup_steps": 2, "num_workers": 0, "embedding_dim": 32,
            "p2_lr": 0.01,
            "p2_lr_decay_epochs": [2],
            "p2_lr_decay_gamma": 0.1,
            "arcface_s": 64.0,
            "arcface_m": 0.5,
        },
        "eval": {"pairs_txt": "unused-by-stub", "max_pairs_inloop": 10, "batch_size": 8},
    }

    result = train_mod.run_training(cfg)

    # 1. Finished and reported some history.
    assert "history" in result
    # 2. Every recorded train loss is finite (no NaN/inf made it into bookkeeping).
    assert all(math.isfinite(v) for v in result["history"]["train_loss"]), \
        f"non-finite loss in history: {result['history']['train_loss']}"
    # 3. With the stubbed probe returning spread=0.12 (>0.05) and accuracy 0.55, the
    #    promotion guard MUST have fired at least once → best_arcface.pt exists.
    best_path = tmp_path / "ckpts" / "best_arcface.pt"
    assert best_path.exists(), f"best_arcface.pt must be produced by guarded best-save, got: {list((tmp_path / 'ckpts').iterdir())}"
    # 4. last.pt always exists (unguarded end-of-training snapshot).
    assert (tmp_path / "ckpts" / "last.pt").exists()
    # 5. ArcFace head was saved separately.
    assert (tmp_path / "ckpts" / "arcface_head.pt").exists()
    # 6. The guarded best checkpoint has the expected schema.
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    assert "model" in ckpt and "cfg" in ckpt
    assert ckpt["cfg"]["checkpoint_name"] == "best_arcface.pt"
