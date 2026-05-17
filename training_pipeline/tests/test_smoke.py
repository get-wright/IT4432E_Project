"""Two-epoch training on synthetic data must reduce loss and save a checkpoint."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from training_pipeline.src.train import run_training


def _make_synthetic_manifest(tmp: Path) -> Path:
    img_dir = tmp / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for ident in range(8):
        base = rng.integers(0, 256, size=3)
        for j in range(8):
            arr = np.clip(base + rng.integers(-20, 20, size=3), 0, 255).astype("uint8")
            img = np.tile(arr, (64, 64, 1))
            p = img_dir / f"id{ident}_{j}.jpg"
            Image.fromarray(img).save(p)
            split = "val" if j < 1 else "train"
            rows.append({
                "path": str(p),
                "identity_id": f"syn_{ident}",
                "source": "syn",
                "split": split,
            })
    df = pd.DataFrame(rows)
    manifest = tmp / "manifest.parquet"
    df.to_parquet(manifest, index=False)
    return manifest


def test_smoke_loss_decreases(tmp_path: Path):
    manifest = _make_synthetic_manifest(tmp_path)
    ckpt_dir = tmp_path / "ckpt"
    tb_dir = tmp_path / "tb"

    cfg = {
        "manifest": str(manifest),
        "checkpoints_dir": str(ckpt_dir),
        "tensorboard_dir": str(tb_dir),
        "seed": 0,
        "train": {
            "epochs": 2, "p": 4, "k": 2, "batches_per_epoch": 5,
            "margin": 0.3, "lr_head": 1e-3, "lr_backbone": 1e-4,
            "weight_decay": 0.0, "warmup_steps": 1, "num_workers": 0,
            "embedding_dim": 64,
        },
        "eval": {"pairs_txt": "", "max_pairs_inloop": 0, "batch_size": 4},
    }
    history = run_training(cfg)
    assert history["train_loss"][0] >= history["train_loss"][-1]
    assert (ckpt_dir / "last.pt").exists()
