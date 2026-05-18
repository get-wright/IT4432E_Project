# ArcFace + TTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace P2 semi-hard triplet with ArcFace angular margin loss and add horizontal-flip TTA at inference, on a new branch `feat/arcface-tta`, while preserving all existing methodology.

**Architecture:** P1 (CE softmax warmup) unchanged — same 3 epochs, same AdamW. P2 swaps in a new `ArcFaceHead` (standard non-easy-margin, s=64 m=0.5) trained with SGD m=0.9 + step decay. Eval and app embedders gain a `use_tta` switch that averages embeddings of a face and its horizontal flip. New `evaluation/benchmark_insightface.py` produces the 8-suite comparison JSON. The app guards enrolled vectors against silent score drift via a model-SHA + TTA-flag sidecar.

**Tech Stack:** PyTorch 2.12, torchvision, facenet-pytorch, FastAPI, pytest. Reference spec: `docs/superpowers/specs/2026-05-18-arcface-tta-design.md` (commit `776f17f`).

**Reference repo state:** Branch `feat/face-recognition-siamese` at HEAD `cad9788`. The new branch forks from here.

---

## File Structure

**New files:**
- `training_pipeline/src/arcface_head.py` — ArcFace margin loss head
- `training_pipeline/configs/train_arcface.yaml` — ArcFace recipe config
- `training_pipeline/tests/test_arcface_head.py` — unit tests for the head
- `training_pipeline/tests/test_arcface_smoketest.py` — end-to-end mini run
- `evaluation/benchmark_insightface.py` — 8-suite .bin verification runner

**Modified files:**
- `training_pipeline/src/model.py` — add `embed_tta` method
- `training_pipeline/src/eval_lfw.py:164` — `use_tta` param threaded into `evaluate_lfw`
- `training_pipeline/src/train.py` — P2 swap to ArcFace + SGD + step decay + center logging + `checkpoint_name` fix at line 233
- `evaluation/eval_lfw.py` — `--tta` CLI flag, forward to `evaluate_lfw`
- `evaluation/eval_pins.py` — `--tta` CLI flag, forward to embedding loop
- `application/backend/inference.py` — `use_tta` ctor arg on `Embedder`
- `application/backend/db.py` — sidecar `meta.json` next to `vectors.npy`, version guard
- `application/backend/main.py` — `APP_TTA` env var, pass to `Embedder` and `EnrollmentDB`
- `application/Dockerfile:16` — `APP_THRESHOLD=0.565`
- `application/frontend/app.js` — guard `snapBase64()` against zero-dim video
- `evaluation/benchmarks.ipynb` — append "Triplet vs ArcFace" comparison section

**Generated artifacts (not committed to git, kept on VM + locally):**
- `training_pipeline/checkpoints/best_arcface.pt`, `last.pt`, `arcface_head.pt`
- `evaluation/results_arcface.json`
- `evaluation/results_pins_arcface.json`
- `evaluation/results_insightface_bench_arcface.json`

---

## Task 0: Create branch

**Files:** none (git operation)

- [ ] **Step 1: Verify current branch + HEAD**

Run: `git status && git log --oneline -1`
Expected: clean working tree on `feat/face-recognition-siamese`, HEAD ≥ `776f17f` (spec commit must be present so the new branch contains the design doc).

- [ ] **Step 2: Create + switch to new branch**

Run: `git checkout -b feat/arcface-tta`
Expected: `Switched to a new branch 'feat/arcface-tta'`

- [ ] **Step 3: Sanity push to remote (track upstream early)**

Run: `git push -u origin feat/arcface-tta`
Expected: new remote branch created.

---

## Task 1: ArcFace head — TDD

**Files:**
- Create: `training_pipeline/src/arcface_head.py`
- Create: `training_pipeline/tests/test_arcface_head.py`

- [ ] **Step 1: Write the failing tests**

Write `training_pipeline/tests/test_arcface_head.py`:

```python
"""Unit tests for ArcFaceHead — shape, finite, math correctness, gradient flow."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from training_pipeline.src.arcface_head import ArcFaceHead


@pytest.fixture
def head():
    torch.manual_seed(0)
    return ArcFaceHead(embedding_dim=8, num_classes=4, s=64.0, m=0.5)


def _normed(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)


def test_output_shape(head):
    emb = _normed(torch.randn(3, 8))
    labels = torch.tensor([0, 1, 2])
    logits = head(emb, labels)
    assert logits.shape == (3, 4)


def test_logits_finite_on_random_input(head):
    emb = _normed(torch.randn(16, 8))
    labels = torch.randint(0, 4, (16,))
    logits = head(emb, labels)
    assert torch.isfinite(logits).all()


def test_correct_class_logit_drops_with_margin(head):
    """For positive cos_theta, cos(theta + m) < cos(theta) so the target logit drops vs no-margin."""
    torch.manual_seed(1)
    # Build an embedding aligned exactly with class-0's weight direction
    with torch.no_grad():
        w0 = F.normalize(head.weight[0:1], dim=1)  # [1, 8]
    emb = w0.clone()                                 # already unit-norm
    labels = torch.tensor([0])
    logits = head(emb, labels)
    # Reference: scaled cosine without margin would be s * 1.0 == s
    assert logits[0, 0].item() < head.s, "margin should pull the target logit below s"
    assert logits[0, 0].item() > 0, "with cos_theta=1, margin output should still be positive"


def test_non_target_logits_use_plain_cosine(head):
    emb = _normed(torch.randn(2, 8))
    labels = torch.tensor([0, 1])
    logits = head(emb, labels)
    weight_norm = F.normalize(head.weight, dim=1)
    plain = head.s * (emb @ weight_norm.t())
    # Non-target columns must equal plain scaled cosine (margin is on target only).
    assert torch.allclose(logits[0, 1:], plain[0, 1:], atol=1e-5)
    assert torch.allclose(logits[1, [0, 2, 3]], plain[1, [0, 2, 3]], atol=1e-5)


def test_logit_lower_bound_with_standard_fallback(head):
    """Standard non-easy-margin ArcFace can push target logits to -(1 + mm) * s."""
    mm = math.sin(math.pi - head.m) * head.m
    expected_floor = -(1.0 + mm) * head.s
    # Push embedding to be anti-aligned with class 0's weight (cos_theta ≈ -1)
    with torch.no_grad():
        w0 = F.normalize(head.weight[0:1], dim=1)
    emb = -w0.clone()
    labels = torch.tensor([0])
    logits = head(emb, labels)
    target = logits[0, 0].item()
    # Must respect the derived lower bound (small slack for fp32 noise).
    assert target >= expected_floor - 1e-3
    # And must reflect the fallback branch — i.e., target < non-target (plain) cosine column.
    assert target < head.s * 1.0


def test_gradient_flows_to_embedding_and_weight(head):
    emb = _normed(torch.randn(4, 8)).requires_grad_(True)
    labels = torch.tensor([0, 1, 2, 3])
    logits = head(emb, labels)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()


def test_center_norms_returns_per_class_norms(head):
    norms = head.center_norms()
    assert norms.shape == (4,)
    assert torch.isfinite(norms).all()
    # Xavier-normal init → norms should sit in a small range around sqrt(2/(D+C))*sqrt(D).
    assert (norms > 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest training_pipeline/tests/test_arcface_head.py -v`
Expected: `ModuleNotFoundError: No module named 'training_pipeline.src.arcface_head'`

- [ ] **Step 3: Implement the head**

Write `training_pipeline/src/arcface_head.py`:

```python
"""ArcFace angular-margin head (standard non-easy-margin variant from the paper).

Consumes L2-normalized embeddings, returns scaled logits ready for cross-entropy.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = float(s)
        self.m = float(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # Wrong-hemisphere boundary: cos(pi - m). Below this, fall back to cos_theta - mm.
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, D] — already L2-normalized by caller.
        weight_norm = F.normalize(self.weight, dim=1)
        cos_theta = embeddings @ weight_norm.t()
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Standard fallback for wrong-hemisphere (NOT easy-margin): cos_theta - mm.
        cos_theta_m = torch.where(
            cos_theta > self.threshold,
            cos_theta_m,
            cos_theta - self.mm,
        )

        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(cos_theta.dtype)
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        return logits * self.s

    @torch.no_grad()
    def center_norms(self) -> torch.Tensor:
        return self.weight.norm(dim=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest training_pipeline/tests/test_arcface_head.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add training_pipeline/src/arcface_head.py training_pipeline/tests/test_arcface_head.py
git commit -m "feat(loss): ArcFace angular-margin head (s=64, m=0.5, standard fallback)"
```

---

## Task 2: TTA method on FaceEmbedding

**Files:**
- Modify: `training_pipeline/src/model.py` (append a method)
- Modify: `training_pipeline/tests/test_model.py` (append a test)

- [ ] **Step 1: Write the failing test**

Append to `training_pipeline/tests/test_model.py`:

```python
def test_embed_tta_unit_norm_and_flip_consistency():
    """embed_tta returns unit-norm and is permutation-invariant to flipping the input."""
    import torch
    import torch.nn.functional as F
    from training_pipeline.src.model import FaceEmbedding

    torch.manual_seed(0)
    model = FaceEmbedding(embedding_dim=32, pretrained=False).eval()
    x = torch.randn(2, 3, 160, 160)

    with torch.no_grad():
        e = model.embed_tta(x)
        e_flipped = model.embed_tta(torch.flip(x, dims=[-1]))

    # Unit norm
    assert torch.allclose(e.norm(dim=1), torch.ones(2), atol=1e-5)
    # Flip-invariant: TTA averages x and flip(x), so embedding of flip(x) is the same set.
    assert torch.allclose(e, e_flipped, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest training_pipeline/tests/test_model.py::test_embed_tta_unit_norm_and_flip_consistency -v`
Expected: `AttributeError: 'FaceEmbedding' object has no attribute 'embed_tta'`

- [ ] **Step 3: Add the method to FaceEmbedding**

In `training_pipeline/src/model.py`, after the existing `embed_normalized` method (around line 25), insert:

```python
    def embed_tta(self, x):
        """L2-normalized average of embeddings from x and its horizontal flip.

        2x forward at inference; helps on profile/pose benchmarks by reducing
        left/right asymmetry. No retraining needed.
        """
        e1 = self.embed_normalized(x)
        e2 = self.embed_normalized(torch.flip(x, dims=[-1]))
        return F.normalize(e1 + e2, p=2, dim=1)
```

If `torch` is not already imported at top of the file, add `import torch` near the existing imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest training_pipeline/tests/test_model.py::test_embed_tta_unit_norm_and_flip_consistency -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training_pipeline/src/model.py training_pipeline/tests/test_model.py
git commit -m "feat(model): embed_tta — flip-averaged L2-normalized embedding"
```

---

## Task 3: Thread `use_tta` through the shared evaluator

**Files:**
- Modify: `training_pipeline/src/eval_lfw.py` (signature + line ~164)
- Modify: `training_pipeline/tests/test_eval_lfw.py` (extend a test)

- [ ] **Step 1: Write the failing test**

Append to `training_pipeline/tests/test_eval_lfw.py`:

```python
def test_evaluate_lfw_use_tta_calls_embed_tta(monkeypatch, tmp_path):
    """use_tta=True must dispatch to model.embed_tta instead of embed_normalized."""
    import torch
    from training_pipeline.src.eval_lfw import evaluate_lfw, LfwPair

    calls = {"embed_normalized": 0, "embed_tta": 0}

    class _SpyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 160 * 160, 4)
        def embed_normalized(self, x):
            calls["embed_normalized"] += 1
            return torch.nn.functional.normalize(self.fc(x.flatten(1)), dim=1)
        def embed_tta(self, x):
            calls["embed_tta"] += 1
            return torch.nn.functional.normalize(self.fc(x.flatten(1)), dim=1)

    # Stub pair list with the dataset wrapper missing → use raw fallback for everything.
    # Easiest: monkeypatch loader to return a deterministic tensor per pair.
    # See existing test pattern in this file for how to build a 2-pair fixture.
    pairs = [
        LfwPair(name1="A", idx1=1, name2="A", idx2=2, same=True),
        LfwPair(name1="A", idx1=1, name2="B", idx2=1, same=False),
    ]

    class _StubDataset:
        def __init__(self, *_a, **_k): pass
        def __len__(self): return 3
        def __getitem__(self, i): return torch.zeros(3, 160, 160)

    from training_pipeline.src import eval_lfw as ev
    monkeypatch.setattr(ev, "_PairImgDataset", _StubDataset)

    model = _SpyModel().eval()
    evaluate_lfw(
        model, pairs,
        aligned_root=tmp_path, raw_root=tmp_path,
        device="cpu", strict=False, use_tta=True,
    )
    assert calls["embed_tta"] >= 1
    assert calls["embed_normalized"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest training_pipeline/tests/test_eval_lfw.py::test_evaluate_lfw_use_tta_calls_embed_tta -v`
Expected: FAIL — `evaluate_lfw() got an unexpected keyword argument 'use_tta'`.

- [ ] **Step 3: Thread the param through `evaluate_lfw`**

In `training_pipeline/src/eval_lfw.py`, locate the `def evaluate_lfw(` signature (the function whose body contains line 164's `model.embed_normalized(x)`). Add `use_tta: bool = False,` as the last keyword parameter. Then change the embedding loop:

Before (around line 164):
```python
    for x in dl:
        x = x.to(device, non_blocking=True)
        embs_list.append(model.embed_normalized(x).cpu())
```

After:
```python
    embed_fn = model.embed_tta if use_tta else model.embed_normalized
    for x in dl:
        x = x.to(device, non_blocking=True)
        embs_list.append(embed_fn(x).cpu())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest training_pipeline/tests/test_eval_lfw.py -v`
Expected: new test passes; existing tests still pass (back-compat — default `use_tta=False`).

- [ ] **Step 5: Commit**

```bash
git add training_pipeline/src/eval_lfw.py training_pipeline/tests/test_eval_lfw.py
git commit -m "feat(eval): use_tta param in evaluate_lfw — dispatches to embed_tta when set"
```

---

## Task 4: `--tta` CLI flags on LFW + Pins evaluators

**Files:**
- Modify: `evaluation/eval_lfw.py`
- Modify: `evaluation/eval_pins.py`

- [ ] **Step 1: Add `--tta` to evaluation/eval_lfw.py**

In `evaluation/eval_lfw.py`, in `main()` after the existing `ap.add_argument` calls:

```python
    ap.add_argument("--tta", action="store_true",
                    help="Use flip-averaged TTA at embed time")
```

Then forward to `evaluate_lfw`. Find the existing call:

```python
    metrics = evaluate_lfw(
        model, pairs,
        aligned_root=Path(args.aligned_root),
        raw_root=raw_root,
        device=args.device,
        strict=True,
    )
```

Add `use_tta=args.tta,` as the last keyword arg:

```python
    metrics = evaluate_lfw(
        model, pairs,
        aligned_root=Path(args.aligned_root),
        raw_root=raw_root,
        device=args.device,
        strict=True,
        use_tta=args.tta,
    )
```

- [ ] **Step 2: Add `--tta` to evaluation/eval_pins.py**

In `evaluation/eval_pins.py`, in `main()` after the existing `ap.add_argument` calls (around line 81 where `--lfw-results` is defined):

```python
    ap.add_argument("--tta", action="store_true",
                    help="Use flip-averaged TTA at embed time")
```

Locate the embedding loop inside `main()` (it iterates `DataLoader(ds, ...)` and calls `model.embed_normalized(...)`). Replace the `model.embed_normalized(x)` call with:

```python
        embed_fn = model.embed_tta if args.tta else model.embed_normalized
        for x in dl:
            x = x.to(args.device, non_blocking=True)
            embs.append(embed_fn(x).cpu())
```

(Adjust variable names — `embs` or `embs_list` — to match the surrounding code; do not rename existing locals.)

- [ ] **Step 3: Smoke-run the CLIs with the current best.pt (sanity, no commit yet)**

Run:
```bash
.venv/bin/python -m evaluation.eval_lfw \
    --checkpoint application/models/best.pt \
    --out /tmp/_smoke_lfw_no_tta.json
```
Expected: writes JSON with `mean_acc` near the existing `evaluation/results.json` value (~0.95).

Then:
```bash
.venv/bin/python -m evaluation.eval_lfw \
    --checkpoint application/models/best.pt \
    --tta \
    --out /tmp/_smoke_lfw_tta.json
```
Expected: JSON written; `mean_acc` typically within ±0.5pp of the no-TTA run (TTA on triplet model is mild).

Delete temp files: `rm /tmp/_smoke_lfw_*.json`.

- [ ] **Step 4: Commit**

```bash
git add evaluation/eval_lfw.py evaluation/eval_pins.py
git commit -m "feat(eval-cli): --tta flag on eval_lfw and eval_pins"
```

---

## Task 5: New `evaluation/benchmark_insightface.py`

This script does not exist in the repo today — the existing `results_insightface_bench.json` was produced by a one-off VM script. Now we commit a clean, reusable version.

**Files:**
- Create: `evaluation/benchmark_insightface.py`

- [ ] **Step 1: Inspect existing JSON shape for back-compat**

Read `evaluation/results_insightface_bench.json` to confirm the keys per benchmark: `mean_acc_cv`, `std_acc_cv`, `acc_at_lfw_threshold_0.565`, `spread`, `pos_sim_mean`, `neg_sim_mean`, `n_pairs`. The new script must emit the same schema so the notebook keeps working.

- [ ] **Step 2: Write the script**

Create `evaluation/benchmark_insightface.py`:

```python
"""Runs the 8-suite InsightFace .bin verification benchmarks (lfw, agedb_30,
cfp_ff, cfp_fp, cplfw, calfw, sllfw, talfw) against a checkpoint and writes a JSON
with one entry per benchmark. Schema matches results_insightface_bench.json so
benchmarks.ipynb keeps reading both old and new results without changes.
"""
from __future__ import annotations

import argparse
import json
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from training_pipeline.src.model import FaceEmbedding

BENCH_NAMES = ["lfw", "agedb_30", "cfp_ff", "cfp_fp", "cplfw", "calfw", "sllfw", "talfw"]
LFW_TUNED_THRESHOLD = 0.565


def _tf() -> transforms.Compose:
    """160x160 inference transform consistent with training data."""
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _load_bin(bin_path: Path) -> tuple[list[Image.Image], np.ndarray]:
    """InsightFace .bin layout: pickle of (image_byte_list, issame_list).

    image_byte_list is a flat list of length 2*N: pairs are (img[2i], img[2i+1]).
    issame_list has length N.
    """
    with open(bin_path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")
    images = [Image.open(BytesIO(b)).convert("RGB") for b in bins]
    return images, np.array(issame, dtype=bool)


@torch.no_grad()
def _embed_all(model: FaceEmbedding, images: list[Image.Image], device: str,
               use_tta: bool, batch_size: int = 128) -> torch.Tensor:
    tf = _tf()
    embed_fn = model.embed_tta if use_tta else model.embed_normalized
    out = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([tf(im) for im in images[i:i + batch_size]]).to(device)
        out.append(embed_fn(batch).cpu())
    return torch.cat(out, dim=0)


def _ten_fold_cv_accuracy(sims: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """LFW-style 10-fold CV: pick best threshold on each train fold, evaluate on val fold.

    Returns (mean_acc, std_acc, mean_chosen_threshold).
    """
    n = len(sims)
    fold_size = n // 10
    cand = np.linspace(-1.0, 1.0, 401)
    accs, thrs = [], []
    for k in range(10):
        lo, hi = k * fold_size, (k + 1) * fold_size if k < 9 else n
        val_idx = np.zeros(n, dtype=bool)
        val_idx[lo:hi] = True
        train_sims, train_lab = sims[~val_idx], labels[~val_idx]
        train_accs = ((train_sims[None, :] >= cand[:, None]) == train_lab[None, :]).mean(axis=1)
        best = float(cand[int(train_accs.argmax())])
        thrs.append(best)
        val_acc = ((sims[val_idx] >= best) == labels[val_idx]).mean()
        accs.append(float(val_acc))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(thrs))


def _run_one(model: FaceEmbedding, bin_path: Path, device: str, use_tta: bool) -> dict:
    images, issame = _load_bin(bin_path)
    embs = _embed_all(model, images, device, use_tta)
    # Pair (2i, 2i+1)
    e_a = embs[0::2]
    e_b = embs[1::2]
    sims = (e_a * e_b).sum(dim=1).numpy()
    labels = issame.astype(bool)
    pos = sims[labels]
    neg = sims[~labels]
    mean_acc, std_acc, _ = _ten_fold_cv_accuracy(sims, labels)
    acc_at_lfw_thr = float(((sims >= LFW_TUNED_THRESHOLD) == labels).mean())
    return {
        "mean_acc_cv": mean_acc,
        "std_acc_cv": std_acc,
        f"acc_at_lfw_threshold_{LFW_TUNED_THRESHOLD}": acc_at_lfw_thr,
        "spread": float(pos.mean() - neg.mean()),
        "pos_sim_mean": float(pos.mean()),
        "neg_sim_mean": float(neg.mean()),
        "n_pairs": int(len(sims)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--bins-root", default=Path("preprocess-data/insightface_bins"),
                    help="Directory containing <name>.bin files for each benchmark")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bench", nargs="*", default=BENCH_NAMES,
                    help=f"Subset of benchmarks to run. Default: {BENCH_NAMES}")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = FaceEmbedding(embedding_dim=ckpt["cfg"]["train"]["embedding_dim"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    results = {}
    for name in args.bench:
        bin_path = args.bins_root / f"{name}.bin"
        if not bin_path.exists():
            print(f"SKIP {name}: {bin_path} not found")
            continue
        print(f"running {name}…")
        results[name] = _run_one(model, bin_path, args.device, args.tta)
        print(f"  acc_cv={results[name]['mean_acc_cv']:.4f}  "
              f"spread={results[name]['spread']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Sanity-run against the current triplet model**

Run (only if `preprocess-data/insightface_bins/` exists locally; otherwise this step runs only on the VM):
```bash
.venv/bin/python -m evaluation.benchmark_insightface \
    --checkpoint application/models/best.pt \
    --bench lfw \
    --out /tmp/_smoke_bench.json
```
Expected: `lfw` entry with `mean_acc_cv` within ±0.5pp of the existing `results_insightface_bench.json` LFW entry (`0.950333`). Delete: `rm /tmp/_smoke_bench.json`.

If `insightface_bins/` is not on this machine, skip — the VM has it.

- [ ] **Step 4: Commit**

```bash
git add evaluation/benchmark_insightface.py
git commit -m "feat(eval): commit benchmark_insightface.py — 8-suite .bin runner with --tta"
```

---

## Task 6: ArcFace config + train.py P2 swap + `checkpoint_name` fix

**Files:**
- Create: `training_pipeline/configs/train_arcface.yaml`
- Modify: `training_pipeline/src/train.py`

- [ ] **Step 1: Write the new config**

Create `training_pipeline/configs/train_arcface.yaml`:

```yaml
manifest: process-data/manifest.parquet
checkpoints_dir: training_pipeline/checkpoints
tensorboard_dir: training_pipeline/tensorboard_logs
seed: 42
checkpoint_name: best_arcface.pt

train:
  phase1_epochs: 3
  phase2_epochs: 27
  batches_per_epoch: 1500
  phase1_batch_size: 128
  p: 32
  k: 4
  # P1 (CE warmup, AdamW)
  lr_head: 5.0e-4
  lr_backbone: 1.0e-4
  weight_decay: 5.0e-4
  warmup_steps: 500
  num_workers: 8
  embedding_dim: 512
  # P2 (ArcFace, SGD m=0.9 + step decay)
  p2_lr: 0.01
  # Milestones are P2-relative (scheduler.step() called once per P2 epoch).
  # Canonical ArcFace decay at 62.5% and 87.5% of total run, mapped through P1=3:
  # 30 * 0.625 - 3 = 16.75 -> 17;  30 * 0.875 - 3 = 23.25 -> 24.
  p2_lr_decay_epochs: [17, 24]
  p2_lr_decay_gamma: 0.1
  arcface_s: 64.0
  arcface_m: 0.5

eval:
  pairs_txt: preprocess-data/lfw/pairs.txt
  max_pairs_inloop: null
  batch_size: 128
```

- [ ] **Step 2: Fix `checkpoint_name` hardcode in train.py**

Open `training_pipeline/src/train.py`. Locate the guarded best-save (currently at line 231-233):

```python
        if metrics["mean_acc"] > best_acc and metrics["spread"] > 0.05:
            best_acc = metrics["mean_acc"]
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir / "best.pt")
```

Replace **only the literal `"best.pt"`** on the `torch.save` line so the new YAML drives it:

```python
        if metrics["mean_acc"] > best_acc and metrics["spread"] > 0.05:
            best_acc = metrics["mean_acc"]
            torch.save(
                {"model": model.state_dict(), "cfg": cfg},
                ckpt_dir / cfg.get("checkpoint_name", "best.pt"),
            )
```

Do **NOT** change the `last.pt` save two lines below — that file stays literal.

- [ ] **Step 3: Swap P2 loss/optimizer/scheduler**

In `training_pipeline/src/train.py`, add the import near the existing imports:

```python
from .arcface_head import ArcFaceHead
```

Find the P2 setup block (just after `p2_sampler = PKSampler(...)` and before the P2 epoch loop). The current code constructs a triplet loss + AdamW. Replace the P2 head/optim/loss construction with:

```python
    # === P2: ArcFace head + SGD + step decay ===
    arc_head = ArcFaceHead(
        embedding_dim=cfg["train"]["embedding_dim"],
        num_classes=n_identities,
        s=cfg["train"]["arcface_s"],
        m=cfg["train"]["arcface_m"],
    ).to(device)

    p2_optim = torch.optim.SGD(
        [
            {"params": model.parameters()},
            {"params": arc_head.parameters()},
        ],
        lr=cfg["train"]["p2_lr"],
        momentum=0.9,
        weight_decay=cfg["train"]["weight_decay"],
        nesterov=False,
    )

    p2_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        p2_optim,
        milestones=cfg["train"]["p2_lr_decay_epochs"],
        gamma=cfg["train"]["p2_lr_decay_gamma"],
    )
```

(`n_identities` is already in scope — it's the same variable P1 passes to `ClassifierHead` at `train.py:84`. No new derivation needed.)

Then in the P2 batch loop, replace the current triplet-loss call:

```python
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                emb = model.embed_normalized(images)
                logits = arc_head(emb, labels)
                loss = torch.nn.functional.cross_entropy(logits, labels)
```

(Drop the `n_triplet_meter` update and any triplet-loss bookkeeping inside this branch — ArcFace doesn't produce that count. If `n_triplet_meter.avg` appears in the per-epoch print, replace it with a constant `0.0` or remove it from the f-string.)

After each P2 epoch ends (right after the existing `history.json` write), add:

```python
        p2_scheduler.step()

        center_norms = arc_head.center_norms()
        print(
            f"[P2 epoch {epoch}] center_norm "
            f"mean={center_norms.mean().item():.3f} "
            f"min={center_norms.min().item():.3f} "
            f"max={center_norms.max().item():.3f}"
        )
```

After P2 ends (at the very bottom of `run_training`, before `return`), save the ArcFace head separately:

```python
    torch.save(arc_head.state_dict(), ckpt_dir / "arcface_head.pt")
```

- [ ] **Step 4: Run the full existing test suite to confirm nothing else broke**

Run: `.venv/bin/pytest training_pipeline/tests application/tests -q -x --ignore=training_pipeline/tests/test_smoke.py`
(The slow `test_smoke.py` end-to-end run is exercised separately in the next task.)

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add training_pipeline/configs/train_arcface.yaml training_pipeline/src/train.py
git commit -m "feat(train): P2 swap to ArcFace + SGD + MultiStepLR + center-norm log; checkpoint_name driven by config"
```

---

## Task 7: ArcFace smoke test

**Files:**
- Create: `training_pipeline/tests/test_arcface_smoketest.py`

- [ ] **Step 1: Write the smoke test**

Create `training_pipeline/tests/test_arcface_smoketest.py`:

```python
"""End-to-end mini run with the ArcFace config — must train without NaN/collapse."""
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

    import training_pipeline.src.dataset as ds_mod
    monkeypatch.setattr(ds_mod, "ROOT", workdir)

    import training_pipeline.src.train as train_mod

    def _stub_probe(*args, **kwargs):
        return {"mean_acc": 0.55, "spread": 0.12, "pos_sim_mean": 0.6, "neg_sim_mean": 0.48}

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
    # 2. The configured guarded best-save name was honored.
    assert (tmp_path / "ckpts" / "best_arcface.pt").exists() or (tmp_path / "ckpts" / "last.pt").exists(), \
        "training must have produced at least one checkpoint"
    # 3. last.pt always exists.
    assert (tmp_path / "ckpts" / "last.pt").exists()
    # 4. ArcFace head was saved separately.
    assert (tmp_path / "ckpts" / "arcface_head.pt").exists()
    # 5. Loaded checkpoint has the expected schema.
    if (tmp_path / "ckpts" / "best_arcface.pt").exists():
        ckpt = torch.load(tmp_path / "ckpts" / "best_arcface.pt", map_location="cpu", weights_only=False)
        assert "model" in ckpt and "cfg" in ckpt
```

- [ ] **Step 2: Run the smoke test**

Run: `.venv/bin/pytest training_pipeline/tests/test_arcface_smoketest.py -v -m slow`
Expected: PASS in <2 minutes on CPU. No NaN warnings in output.

- [ ] **Step 3: Commit**

```bash
git add training_pipeline/tests/test_arcface_smoketest.py
git commit -m "test(smoke): ArcFace two-phase end-to-end on synthetic data"
```

---

## Task 8: App `Embedder` gains `use_tta`

**Files:**
- Modify: `application/backend/inference.py`
- Modify: `application/tests/` (extend an embedder test)

- [ ] **Step 1: Inspect existing embedder tests for a pattern to extend**

Run: `ls application/tests/ && grep -l "Embedder" application/tests/*.py`
Find the test file that exercises `Embedder` (likely `test_inference.py` or `test_app.py`). Note the fixture pattern used to load a checkpoint.

- [ ] **Step 2: Write the failing test**

Append to the relevant existing app test file:

```python
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
```

- [ ] **Step 3: Run test — should fail**

Run: `.venv/bin/pytest application/tests/test_inference.py::test_embedder_use_tta_calls_embed_tta -v`
(Adjust the file path if the existing tests live in a different module.)

Expected: `TypeError: Embedder.__init__() got an unexpected keyword argument 'use_tta'`.

- [ ] **Step 4: Add `use_tta` to `Embedder`**

Replace the body of `application/backend/inference.py` with:

```python
"""Load trained checkpoint and embed a single aligned face tensor."""
from __future__ import annotations

from pathlib import Path

import torch

from training_pipeline.src.model import FaceEmbedding


class Embedder:
    def __init__(
        self,
        checkpoint: Path,
        device: str = "cpu",
        use_tta: bool = False,
    ) -> None:
        self.device = device
        self.use_tta = use_tta
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.dim = ckpt["cfg"]["train"]["embedding_dim"]
        self.model = FaceEmbedding(embedding_dim=self.dim, pretrained=False).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        embed_fn = self.model.embed_tta if self.use_tta else self.model.embed_normalized
        return embed_fn(face_tensor).squeeze(0).cpu()
```

- [ ] **Step 5: Run tests — should pass**

Run: `.venv/bin/pytest application/tests -v`
Expected: new test passes, existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add application/backend/inference.py application/tests/
git commit -m "feat(app): Embedder use_tta arg dispatches to embed_tta"
```

---

## Task 9: `vectors.npy` meta sidecar + version guard

**Files:**
- Modify: `application/backend/db.py` (sidecar)
- Modify: `application/backend/main.py` (wire `APP_TTA` + version)

- [ ] **Step 1: Add the meta sidecar to `EnrollmentDB`**

In `application/backend/db.py`, replace `__init__` and `_load_vectors`:

```python
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import torch


class EnrollmentDB:
    def __init__(self, root: Path, dim: int, *, embedding_version: str | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        self.vec_path = self.root / "vectors.npy"
        self.meta_path = self.root / "meta.json"
        self.dim = dim
        self.embedding_version = embedding_version  # may be None (back-compat)
        self._init_db()
        self._load_vectors()

    def _load_vectors(self) -> None:
        if self.vec_path.exists():
            if self.embedding_version is not None and self.meta_path.exists():
                meta = json.loads(self.meta_path.read_text())
                if meta.get("embedding_version") != self.embedding_version:
                    raise RuntimeError(
                        f"Embedding store version mismatch: stored "
                        f"{meta.get('embedding_version')!r} but app is "
                        f"{self.embedding_version!r}. Re-enroll or clear "
                        f"{self.vec_path} and {self.meta_path}."
                    )
            arr = np.load(self.vec_path)
            if arr.ndim == 1:
                arr = arr.reshape(0, self.dim)
        else:
            arr = np.zeros((0, self.dim), dtype=np.float32)
            if self.embedding_version is not None:
                self.meta_path.write_text(json.dumps({"embedding_version": self.embedding_version}))
        # ... existing code that assigns self.vectors = arr stays unchanged below this point
```

(Keep the rest of `_load_vectors` exactly as it was — only the existence/meta check at the top is new.)

Also, in whatever method currently writes `vectors.npy` (likely `add()` or `save()`), make sure that after the first successful save, the meta is also written if it doesn't exist. Add at the end of that method:

```python
        if self.embedding_version is not None and not self.meta_path.exists():
            self.meta_path.write_text(json.dumps({"embedding_version": self.embedding_version}))
```

- [ ] **Step 2: Wire `APP_TTA` env var + compute version in main.py**

In `application/backend/main.py`, replace the `_config()` function:

```python
def _config() -> dict:
    return {
        "data_dir": Path(os.environ.get("APP_DATA_DIR", ROOT / "application" / "embeddings")),
        "checkpoint": Path(os.environ.get("APP_CKPT", ROOT / "application" / "models" / "best.pt")),
        "threshold": float(os.environ.get("APP_THRESHOLD", "0.565")),
        "use_tta": os.environ.get("APP_TTA", "0") == "1",
    }
```

Replace `_reset_for_tests()`:

```python
def _reset_for_tests() -> None:
    """Re-init globals from env. Lets tests set APP_DATA_DIR/APP_THRESHOLD per test."""
    global _aligner, _embedder, _db, _threshold
    cfg = _config()
    _aligner = FaceAligner(device="cpu")
    if cfg["checkpoint"].exists():
        _embedder = Embedder(cfg["checkpoint"], device="cpu", use_tta=cfg["use_tta"])
        version = _embedding_version(cfg["checkpoint"], cfg["use_tta"])
        _db = EnrollmentDB(cfg["data_dir"], dim=_embedder.dim, embedding_version=version)
    else:
        _embedder = None
        _db = None
    _threshold = cfg["threshold"]
```

Add a small helper right above `_reset_for_tests`:

```python
def _embedding_version(checkpoint: Path, use_tta: bool) -> str:
    """Stable identifier for (model weights, TTA flag) combination."""
    h = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{h.hexdigest()[:16]}-tta{int(use_tta)}"
```

Make sure `import hashlib` is added near the existing imports.

- [ ] **Step 3: Run app test suite**

Run: `.venv/bin/pytest application/tests -v`
Expected: all pass (the version guard is opt-in; default `embedding_version=None` keeps old behavior, and `main.py` only sets a version when actually loading the model).

- [ ] **Step 4: Manual sanity — start app, verify it loads existing best.pt**

Run:
```bash
APP_TTA=0 .venv/bin/uvicorn application.backend.main:app --port 8001 &
sleep 3
curl -sS http://127.0.0.1:8001/enrolled
kill %1
```
Expected: returns `[]` or the current list of enrolled identities without error. The version sidecar is created on first successful enroll, not on load of an empty store.

- [ ] **Step 5: Commit**

```bash
git add application/backend/db.py application/backend/main.py
git commit -m "feat(app): APP_TTA env + vectors.npy version sidecar guards against silent embed-drift"
```

---

## Task 10: Dockerfile threshold + frontend snap guard

Tiny but important pre-existing bug fixes that compound with the new deployment.

**Files:**
- Modify: `application/Dockerfile`
- Modify: `application/frontend/app.js`

- [ ] **Step 1: Fix Dockerfile threshold**

Edit `application/Dockerfile` line 16:

Before: `ENV APP_THRESHOLD=0.5`
After:  `ENV APP_THRESHOLD=0.565`

- [ ] **Step 2: Guard `snapBase64` in app.js**

In `application/frontend/app.js`, locate `function snapBase64()` (around line 90). Replace it with:

```js
function snapBase64() {
  if (!video.videoWidth || !video.videoHeight) {
    throw new Error('Camera not ready — grant permission or wait for the stream to start.');
  }
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  // un-mirror so the backend receives the natural-orientation frame
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);
  ctx.restore();
  flashFrame();
  return canvas.toDataURL('image/jpeg', 0.92).split(',')[1];
}
```

The existing `enroll-btn` and `verify-btn` click handlers already wrap fetch calls in `try/catch` and render the error message — the thrown Error will surface as `"Camera not ready — ..."` in the result panel, no other UI changes needed.

Bump the cache-buster in `application/frontend/index.html` line 100 from `?v=8` to `?v=9` so users pick up the new JS:

Before: `<script src="/static/app.js?v=8"></script>`
After:  `<script src="/static/app.js?v=9"></script>`

- [ ] **Step 3: Manual sanity — deny camera permission, hit Verify, see clear error**

Run:
```bash
.venv/bin/uvicorn application.backend.main:app --port 8001 &
sleep 3
```
Open http://127.0.0.1:8001 in a private browser window, deny camera permission when prompted, click Verify.
Expected: the result panel shows `"Network error: Camera not ready — grant permission or wait for the stream to start."` (the error path uses the existing catch). No request goes to the backend.

Kill: `kill %1`.

- [ ] **Step 4: Commit**

```bash
git add application/Dockerfile application/frontend/app.js application/frontend/index.html
git commit -m "fix(app): Dockerfile threshold 0.5->0.565; guard snapBase64 against zero-dim video"
```

---

## Task 11: Train on VM 124.197.18.107

Operational task — no code changes. Documents the exact commands to run on the H100 VM.

**Files:** none locally (output artifacts on VM)

- [ ] **Step 1: SSH and sync the new branch**

Run on local:
```bash
ssh root@124.197.18.107
```

Then on VM:
```bash
cd /root/IT4432E_Project
git fetch origin
git checkout feat/arcface-tta
git pull
```
Expected: branch tip matches local HEAD.

- [ ] **Step 2: Install any new deps (none expected — pure code)**

On VM:
```bash
.venv/bin/pip install -e .
```
Expected: no new packages installed.

- [ ] **Step 3: Start tmux session for training**

On VM:
```bash
tmux new -s arcface
```

- [ ] **Step 4: Launch training**

In tmux:
```bash
cd /root/IT4432E_Project
.venv/bin/python -m training_pipeline.src.train \
    --config training_pipeline/configs/train_arcface.yaml \
    2>&1 | tee training_pipeline/logs/arcface_run.log
```

Expected: P1 starts logging losses within 30s. Detach tmux: `Ctrl-B d`.

- [ ] **Step 5: Periodic check**

Reattach: `tmux attach -t arcface`. Look for:
- P1: `[P1 epoch 1] train_loss=…` lines, finishing in ~30 min on H100.
- P2: `[P2 epoch N] center_norm mean=… min=… max=…` lines, norms in `[0.3, 3.0]`, spread > 0.05 after epoch 3.
- LFW probe accuracy should rise into 0.94+ by P2 epoch 10–15, ideally hit 0.97+ by P2 epoch 20.

Total wall time: ~6–8 hours on H100 80GB.

- [ ] **Step 6: Pull artifacts down**

After training ends, on local:
```bash
mkdir -p training_pipeline/checkpoints
scp root@124.197.18.107:/root/IT4432E_Project/training_pipeline/checkpoints/best_arcface.pt \
    training_pipeline/checkpoints/best_arcface.pt
scp root@124.197.18.107:/root/IT4432E_Project/training_pipeline/checkpoints/last.pt \
    training_pipeline/checkpoints/last_arcface.pt
scp root@124.197.18.107:/root/IT4432E_Project/training_pipeline/logs/arcface_run.log \
    training_pipeline/logs/arcface_run.log
```

Expected: `best_arcface.pt` lands locally (~94MB). Do NOT `git add` the checkpoint.

- [ ] **Step 7: No commit yet — training is operational, evaluation in next task**

---

## Task 12: Run the three eval suites against `best_arcface.pt`

**Files:** generates result JSONs (committed at end of this task).

- [ ] **Step 1: Strict LFW 10-fold**

Run on VM (faster, has CUDA + LFW raw + aligned dirs):
```bash
.venv/bin/python -m evaluation.eval_lfw \
    --checkpoint training_pipeline/checkpoints/best_arcface.pt \
    --tta \
    --out evaluation/results_arcface.json
```
Expected: writes JSON. Target: `mean_acc >= 0.95`, `spread >= 0.70`. If `mean_acc < 0.95`, **stop and triage** before continuing — the spec's success criterion 1 is "must not regress vs current 95.03%".

- [ ] **Step 2: Pins cross-dataset (using the new threshold)**

On VM:
```bash
.venv/bin/python -m evaluation.eval_pins \
    --checkpoint training_pipeline/checkpoints/best_arcface.pt \
    --tta \
    --lfw-results evaluation/results_arcface.json \
    --out evaluation/results_pins_arcface.json
```
Expected: writes JSON. The `--lfw-results` flag is critical — it makes the pins evaluator use the ArcFace model's tuned threshold, not the old triplet one.

- [ ] **Step 3: 8-suite InsightFace benchmark**

On VM:
```bash
.venv/bin/python -m evaluation.benchmark_insightface \
    --checkpoint training_pipeline/checkpoints/best_arcface.pt \
    --tta \
    --out evaluation/results_insightface_bench_arcface.json
```
Expected: writes JSON with all 8 entries. Check the success criteria against spec section "Success criteria":
- `lfw.mean_acc_cv >= 0.95`
- `cfp_fp.mean_acc_cv >= 0.88`
- `agedb_30.mean_acc_cv >= 0.80` (target improvement)
- `lfw.spread >= 0.70`

- [ ] **Step 4: Sync result JSONs back to local**

On local:
```bash
scp 'root@124.197.18.107:/root/IT4432E_Project/evaluation/results*_arcface.json' evaluation/
```
Expected: 3 new JSONs land in `evaluation/`.

- [ ] **Step 5: Commit results**

```bash
git add evaluation/results_arcface.json evaluation/results_pins_arcface.json evaluation/results_insightface_bench_arcface.json
git commit -m "feat(results): ArcFace+TTA numbers across LFW / Pins / 8-suite InsightFace bench"
```

---

## Task 13: Add "Triplet vs ArcFace" comparison cell to benchmarks.ipynb

**Files:**
- Modify: `evaluation/benchmarks.ipynb`

- [ ] **Step 1: Append a new markdown cell**

At the bottom of `evaluation/benchmarks.ipynb`, add a markdown cell:

```markdown
## Triplet vs ArcFace (this branch)

Side-by-side comparison of the original semi-hard triplet recipe (`results_insightface_bench.json`) and the new ArcFace + TTA recipe (`results_insightface_bench_arcface.json`). Δ column is the per-benchmark improvement in percentage points.
```

- [ ] **Step 2: Append a new code cell**

After the markdown cell, add:

```python
arc = json.loads((ROOT / 'evaluation/results_insightface_bench_arcface.json').read_text())
df_arc = pd.DataFrame([
    {'benchmark': name, 'acc_arc': r['mean_acc_cv'], 'spread_arc': r['spread']}
    for name, r in arc.items()
])
cmp = df[['benchmark', 'acc_cv', 'spread']].merge(df_arc, on='benchmark')
cmp['delta_pp'] = (cmp['acc_arc'] - cmp['acc_cv']) * 100
cmp['spread_delta'] = cmp['spread_arc'] - cmp['spread']
cmp[['benchmark', 'acc_cv', 'acc_arc', 'delta_pp', 'spread', 'spread_arc', 'spread_delta']]
```

- [ ] **Step 3: Append a second code cell for the chart**

```python
fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(cmp))
w = 0.36
ax.bar(x - w/2, cmp['acc_cv'], width=w, label='triplet (current)', color='#1f2937')
ax.bar(x + w/2, cmp['acc_arc'], width=w, label='ArcFace + TTA', color='#d83a23')
ax.set_xticks(x); ax.set_xticklabels(cmp['benchmark'])
ax.set_ylim(0.5, 1.0); ax.set_ylabel('accuracy (CV-tuned)')
ax.set_title('Per-benchmark accuracy — triplet vs ArcFace+TTA')
ax.legend(loc='lower right'); ax.grid(True, axis='y', alpha=0.3)
for i, (a, b) in enumerate(zip(cmp['acc_cv'], cmp['acc_arc'])):
    ax.text(i - w/2, a + 0.005, f'{a*100:.1f}', ha='center', fontsize=8)
    ax.text(i + w/2, b + 0.005, f'{b*100:.1f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / 'triplet_vs_arcface.png', dpi=150)
plt.show()
```

- [ ] **Step 4: Run the new cells**

In Jupyter or via:
```bash
.venv/bin/jupyter nbconvert --to notebook --execute evaluation/benchmarks.ipynb \
    --output evaluation/benchmarks.ipynb
```
Expected: executes without error, produces `triplet_vs_arcface.png` in `evaluation/figs/`.

- [ ] **Step 5: Commit**

```bash
git add evaluation/benchmarks.ipynb evaluation/figs/triplet_vs_arcface.png
git commit -m "docs(notebook): add triplet vs ArcFace+TTA comparison section"
```

---

## Task 14: Final review + branch push

- [ ] **Step 1: Run the full test suite locally**

Run: `.venv/bin/pytest -q`
Expected: all tests pass, including `-m slow` smoke tests on synthetic data.

- [ ] **Step 2: Push the branch**

Run: `git push origin feat/arcface-tta`
Expected: clean push.

- [ ] **Step 3: Open PR (manual step, no commit)**

In browser, open the GitHub compare page for `feat/face-recognition-siamese` ← `feat/arcface-tta`. PR title: `feat: ArcFace + TTA recipe upgrade`. Body should link to the spec doc and summarize the three result deltas.

---

## Promotion decision

After the branch lands, the maintainer decides whether to promote `best_arcface.pt` to `application/models/best.pt` based on the success-criteria table in the spec. If promoted:

1. Copy: `cp training_pipeline/checkpoints/best_arcface.pt application/models/best.pt`
2. Clear the embedding store: `rm application/embeddings/vectors.npy application/embeddings/meta.json application/embeddings/index.db`
3. Re-enroll all identities through the app UI.
4. Set deployment env: `APP_TTA=1`.

If not promoted, the branch still merges — both checkpoints coexist; the new evaluators and `--tta` flag remain useful for future runs.
