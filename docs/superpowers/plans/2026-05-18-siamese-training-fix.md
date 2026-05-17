# Siamese Training Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the collapsed triplet training with the canonical softmax-warmup → semi-hard triplet recipe, repair the LFW eval so it can never silently leak again, and verify end-to-end that LFW + Pins both show real discrimination.

**Architecture:** Two-phase training. Phase 1 (3 epochs) trains FaceEmbedding + a temporary classifier head with cross-entropy on identity labels — guarantees non-degenerate embeddings before mining starts. Phase 2 (17 epochs) drops the classifier, switches to PKSampler + semi-hard negative mining with soft-margin loss on L2-normalized embeddings. Eval pipeline keeps all 6000 LFW pairs (raw fallback when alignment fails), enforces distribution + threshold sanity assertions, and adds a fixed-threshold Pins cross-dataset check.

**Tech Stack:** PyTorch 2.x, torchvision (ResNet50), facenet-pytorch (MTCNN), pytest, YAML config, AdamW + mixed precision on H100.

**Spec:** `docs/superpowers/specs/2026-05-18-siamese-training-fix-design.md`

---

## File map

**Modify:**
- `training_pipeline/src/model.py` — un-normalized forward, add `embed_normalized()`, add `ClassifierHead` module.
- `training_pipeline/src/loss.py` — add `softplus_loss`, `semi_hard_triplet_loss`; keep `batch_hard_triplet_loss` for ablation.
- `training_pipeline/src/dataset.py` — drop `RandomErasing` from `train_transform()`.
- `training_pipeline/src/eval_lfw.py` — `LfwPair` dataclass, no-drop loader, raw fallback, strict/non-strict assertions split.
- `training_pipeline/src/train.py` — two-phase rewrite.
- `training_pipeline/configs/train.yaml` — new hyperparams (phase split, LRs, batch sizes).
- `evaluation/eval_lfw.py` — CLI wires new `evaluate_lfw` signature, writes extended metrics JSON.
- `training_pipeline/tests/test_model.py` — add forward/embed_normalized/classifier tests.
- `training_pipeline/tests/test_loss.py` — add semi-hard and softplus tests.
- `training_pipeline/tests/test_smoke.py` — adapt to two-phase.
- `process-data/process.py` — call new `lfw_layout.find_lfw_identity_root` (preserve current behaviour).
- `README.md` — replace bogus 97.48% with real numbers after training.

**Create:**
- `process-data/lfw_layout.py` — `find_lfw_identity_root(raw_dir)` discovery helper.
- `evaluation/eval_pins.py` — fixed-threshold Pins cross-dataset evaluator.
- `training_pipeline/tests/test_lfw_layout.py`.
- `training_pipeline/tests/test_eval_lfw.py`.
- `evaluation/results.json` (overwritten by training; tracked as artifact).
- `evaluation/results_pins.json` (artifact).

---

## Task ordering

1. lfw_layout (independent helper)
2. process.py refactor (uses lfw_layout)
3. dataset.py (drop RandomErasing)
4. model.py changes
5. loss.py additions
6. eval_lfw.py rewrite
7. evaluation/eval_lfw.py CLI
8. evaluation/eval_pins.py
9. train.py two-phase
10. configs/train.yaml
11. test_smoke.py update
12. Local pytest gate
13. Push to VM, train, monitor
14. Pull artifacts, run final evals, commit results
15. Update README + notebooks

---

### Task 1: LFW layout discovery helper

**Files:**
- Create: `process-data/lfw_layout.py`
- Test: `training_pipeline/tests/test_lfw_layout.py`

- [ ] **Step 1: Write the failing tests**

```python
# training_pipeline/tests/test_lfw_layout.py
import sys
from pathlib import Path

import pytest

# process-data has a hyphen so we have to add it to sys.path explicitly.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "process-data"))

from lfw_layout import find_lfw_identity_root  # noqa: E402


def _make_identity(parent: Path, name: str = "Person_A") -> None:
    person = parent / name
    person.mkdir(parents=True)
    (person / f"{name}_0001.jpg").write_bytes(b"")


def test_find_root_flat(tmp_path):
    _make_identity(tmp_path)
    assert find_lfw_identity_root(tmp_path) == tmp_path


def test_find_root_single_nested(tmp_path):
    inner = tmp_path / "lfw-deepfunneled"
    _make_identity(inner)
    assert find_lfw_identity_root(tmp_path) == inner


def test_find_root_double_nested(tmp_path):
    inner = tmp_path / "lfw-deepfunneled" / "lfw-deepfunneled"
    _make_identity(inner)
    assert find_lfw_identity_root(tmp_path) == inner


def test_find_root_funneled_variant(tmp_path):
    inner = tmp_path / "lfw_funneled"
    _make_identity(inner)
    assert find_lfw_identity_root(tmp_path) == inner


def test_find_root_no_identities_raises(tmp_path):
    (tmp_path / "some-random-file.txt").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        find_lfw_identity_root(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest training_pipeline/tests/test_lfw_layout.py -v
```

Expected: ImportError / ModuleNotFoundError on `lfw_layout`.

- [ ] **Step 3: Implement `process-data/lfw_layout.py`**

```python
"""Discover the directory containing LFW identity folders.

The Kaggle LFW dumps ship with inconsistent nesting:
- flat:                <root>/<Person>/<Person>_0001.jpg
- single-nested:       <root>/lfw-deepfunneled/<Person>/...
- double-nested:       <root>/lfw-deepfunneled/lfw-deepfunneled/<Person>/...
- funneled variant:    <root>/lfw_funneled/<Person>/...

`find_lfw_identity_root` walks down from `raw_dir` until it finds a directory
whose children look like identity folders (a subdir containing `<name>_NNNN.jpg`).
"""
from __future__ import annotations

from pathlib import Path


def _looks_like_identity_dir(d: Path) -> bool:
    """A directory whose name matches its file prefix, e.g. 'Aaron_Eckhart' contains 'Aaron_Eckhart_0001.jpg'."""
    if not d.is_dir():
        return False
    for f in d.iterdir():
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if f.name.startswith(d.name + "_"):
                return True
    return False


def find_lfw_identity_root(raw_dir: Path) -> Path:
    """Find the directory that directly contains LFW identity folders.

    Walks down at most 3 levels of nesting. Raises FileNotFoundError if no
    identity-shaped directory is found.
    """
    raw_dir = Path(raw_dir)
    candidates = [raw_dir]
    for depth in range(3):
        next_candidates: list[Path] = []
        for c in candidates:
            if not c.is_dir():
                continue
            for sub in sorted(c.iterdir()):
                if _looks_like_identity_dir(sub):
                    return c
            next_candidates.extend(s for s in c.iterdir() if s.is_dir())
        candidates = next_candidates
        if not candidates:
            break
    raise FileNotFoundError(f"No LFW identity folders found under {raw_dir}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest training_pipeline/tests/test_lfw_layout.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add process-data/lfw_layout.py training_pipeline/tests/test_lfw_layout.py
git commit -m "feat(data): add LFW layout discovery helper"
```

---

### Task 2: Refactor process.py to use lfw_layout

**Files:**
- Modify: `process-data/process.py` (the `build_pairs_lfw` function and any in-line LFW path walking)

- [ ] **Step 1: Read the current LFW path walking logic**

```bash
grep -n "lfw" process-data/process.py
```

Identify the block that walks the LFW dump to find identity folders. It currently handles double-nesting inline.

- [ ] **Step 2: Replace with `find_lfw_identity_root`**

In `process-data/process.py`, near the top with other imports:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent))
from lfw_layout import find_lfw_identity_root
```

Then in `build_pairs_lfw` (or wherever LFW root is discovered):

```python
def build_pairs_lfw() -> list[tuple[Path, Path]]:
    raw_dir = ROOT / "preprocess-data" / "lfw"
    identity_root = find_lfw_identity_root(raw_dir)
    pairs: list[tuple[Path, Path]] = []
    for person_dir in sorted(identity_root.iterdir()):
        if not person_dir.is_dir():
            continue
        for src in sorted(person_dir.glob("*.jpg")):
            out = ROOT / "process-data" / "lfw_pairs" / person_dir.name / src.name
            pairs.append((src, out))
    return pairs
```

(Adapt to match the existing function signature — the key is replacing the inline nesting walk with `find_lfw_identity_root`.)

- [ ] **Step 3: Smoke-test the alignment pipeline can still discover LFW**

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'process-data')
from lfw_layout import find_lfw_identity_root
# This will FileNotFoundError locally if preprocess-data/lfw isn't present — that's fine for the smoke test.
try:
    root = find_lfw_identity_root(Path('preprocess-data/lfw'))
    print(f'OK: {root}')
except FileNotFoundError as e:
    print(f'(no LFW locally — only fail this if run on VM) {e}')
"
```

Expected: prints a path (on VM) or a `(no LFW locally)` line (on laptop). Either is fine — failure would be a TypeError or ImportError.

- [ ] **Step 4: Commit**

```bash
git add process-data/process.py
git commit -m "refactor(data): use lfw_layout helper in process.py"
```

---

### Task 3: Drop RandomErasing from train_transform

**Files:**
- Modify: `training_pipeline/src/dataset.py` (around lines 20-28 — `train_transform`)
- Test: `training_pipeline/tests/test_dataset.py` (add new test)

- [ ] **Step 1: Write a failing test**

Add to `training_pipeline/tests/test_dataset.py`:

```python
def test_train_transform_has_no_random_erasing():
    from torchvision import transforms
    from training_pipeline.src.dataset import train_transform

    tf = train_transform()
    assert isinstance(tf, transforms.Compose)
    names = [type(t).__name__ for t in tf.transforms]
    assert "RandomErasing" not in names, (
        f"RandomErasing is in train_transform: {names}. "
        f"It erases identity regions on 160x160 face crops — drop it."
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest training_pipeline/tests/test_dataset.py::test_train_transform_has_no_random_erasing -v
```

Expected: FAIL with the assertion message.

- [ ] **Step 3: Remove RandomErasing**

In `training_pipeline/src/dataset.py`, change `train_transform`:

```python
def train_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest training_pipeline/tests/test_dataset.py -v
```

Expected: all dataset tests pass.

- [ ] **Step 5: Commit**

```bash
git add training_pipeline/src/dataset.py training_pipeline/tests/test_dataset.py
git commit -m "feat(dataset): drop RandomErasing from train_transform (erases identity on 160x160 face crops)"
```

---

### Task 4: Model — un-normalized forward, embed_normalized, ClassifierHead

**Files:**
- Modify: `training_pipeline/src/model.py`
- Test: `training_pipeline/tests/test_model.py`

- [ ] **Step 1: Write failing tests**

Add to `training_pipeline/tests/test_model.py`:

```python
import torch

from training_pipeline.src.model import FaceEmbedding, ClassifierHead


def test_forward_unnormalized():
    """forward returns un-normalized vectors — required so the classifier head sees real logits and the
    triplet loss can normalize internally (preventing norm-scale escape)."""
    model = FaceEmbedding(embedding_dim=64, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 160, 160)
    with torch.no_grad():
        out = model(x)
    norms = out.norm(p=2, dim=1)
    assert out.shape == (4, 64)
    # If model were L2-normalized, norms would all be ~1.0. We want UN-normalized.
    assert not torch.allclose(norms, torch.ones(4), atol=0.1), (
        f"forward output looks L2-normalized (norms={norms.tolist()}); should be raw."
    )


def test_embed_normalized_unit_vectors():
    model = FaceEmbedding(embedding_dim=64, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 160, 160)
    with torch.no_grad():
        out = model.embed_normalized(x)
    norms = out.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"norms={norms.tolist()}"


def test_classifier_head_shapes():
    head = ClassifierHead(embedding_dim=64, n_identities=10)
    emb = torch.randn(4, 64)
    logits = head(emb)
    assert logits.shape == (4, 10)


def test_classifier_head_has_no_bias():
    """No-bias keeps the per-class weight vectors comparable in norm — standard for face CE warmup."""
    head = ClassifierHead(embedding_dim=64, n_identities=10)
    assert head.fc.bias is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest training_pipeline/tests/test_model.py -v -k "unnormalized or embed_normalized or classifier"
```

Expected: failures — ClassifierHead missing; `test_forward_unnormalized` fails because the current forward L2-normalizes.

- [ ] **Step 3: Update model.py**

Replace the contents of `training_pipeline/src/model.py`:

```python
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FaceEmbedding(nn.Module):
    """ResNet50 backbone + Linear(2048, embedding_dim) head.

    `forward` returns the raw 512-d vector (un-normalized). Use `embed_normalized`
    when you need a unit-norm embedding (eval, app inference). The Phase 2 triplet
    loss normalizes internally; the Phase 1 classifier head consumes the raw vector.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, embedding_dim)

    def forward(self, x):
        return self.backbone(x)

    def embed_normalized(self, x):
        return F.normalize(self.forward(x), p=2, dim=1)


class ClassifierHead(nn.Module):
    """Linear classifier over identity labels for Phase 1 softmax warmup.

    No bias: per-class weight vectors should be comparable in norm — standard
    for face-recognition CE warmup recipes.
    """

    def __init__(self, embedding_dim: int, n_identities: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, n_identities, bias=False)

    def forward(self, emb):
        return self.fc(emb)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest training_pipeline/tests/test_model.py -v
```

Expected: all model tests pass.

- [ ] **Step 5: Update app inference to use `embed_normalized`**

`application/backend/inference.py::Embedder.embed` currently calls `self.model(face_tensor)`. With the un-normalized forward this would break cosine-similarity verification in the DB. Patch:

```python
# application/backend/inference.py — only the embed method changes
@torch.no_grad()
def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
    if face_tensor.dim() == 3:
        face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(self.device)
    return self.model.embed_normalized(face_tensor).squeeze(0).cpu()
```

- [ ] **Step 6: Verify app still imports cleanly**

```bash
python -c "from application.backend.inference import Embedder; print('ok')"
```

Expected: `ok`. (Loading a checkpoint isn't required here — just import sanity.)

- [ ] **Step 7: Commit**

```bash
git add training_pipeline/src/model.py training_pipeline/tests/test_model.py application/backend/inference.py
git commit -m "feat(model): un-normalized forward + embed_normalized + ClassifierHead; app uses embed_normalized"
```

---

### Task 5: Loss — softplus_loss + semi_hard_triplet_loss

**Files:**
- Modify: `training_pipeline/src/loss.py`
- Test: `training_pipeline/tests/test_loss.py`

- [ ] **Step 1: Write failing tests**

Add to `training_pipeline/tests/test_loss.py`:

```python
import math

import torch

from training_pipeline.src.loss import (
    semi_hard_triplet_loss,
    softplus_loss,
)


def test_softplus_loss_positive_when_d_ap_greater():
    d_ap = torch.tensor([0.5])
    d_an = torch.tensor([0.4])
    loss = softplus_loss(d_ap, d_an)
    # softplus(0.5 - 0.4) = log(1 + e^0.1) ≈ 0.744
    assert math.isclose(loss.item(), 0.7444, abs_tol=1e-3)


def test_softplus_loss_smooth_gradient_past_margin():
    """Unlike hard hinge, softplus gives a nonzero gradient even when d_an >> d_ap."""
    d_ap = torch.tensor([0.1], requires_grad=True)
    d_an = torch.tensor([0.9])
    loss = softplus_loss(d_ap, d_an)
    loss.backward()
    assert d_ap.grad is not None
    assert torch.isfinite(d_ap.grad).all()
    assert d_ap.grad.item() > 0  # increasing d_ap raises loss


def test_semi_hard_returns_tuple():
    emb = torch.randn(8, 32)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    result = semi_hard_triplet_loss(emb, labels)
    assert isinstance(result, tuple)
    assert len(result) == 2
    loss, n_triplets = result
    assert isinstance(loss, torch.Tensor)
    assert isinstance(n_triplets, int)


def test_semi_hard_normalizes_input():
    """Distances must be computed on unit-normalized vectors regardless of input norm."""
    # Two scaled copies of the same embedding pattern. After L2-norm they're identical.
    # If the loss doesn't normalize internally, distances differ wildly between the two batches.
    base = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])

    loss_small, n_small = semi_hard_triplet_loss(base, labels, margin=0.3)
    loss_big,   n_big   = semi_hard_triplet_loss(base * 100, labels, margin=0.3)

    # After internal normalization the two should be identical.
    assert n_small == n_big
    assert torch.allclose(loss_small, loss_big, atol=1e-5)


def test_semi_hard_no_valid_returns_graph_connected_zero():
    """All embeddings identical → no negative satisfies d_an > d_ap. Must return n=0 and a
    loss that can have .backward() called safely."""
    emb = torch.ones(4, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss, n_triplets = semi_hard_triplet_loss(emb, labels, margin=0.3)
    assert n_triplets == 0
    assert loss.item() == 0.0
    # Should not error and should not produce NaN gradients.
    loss.backward()
    assert torch.isfinite(emb.grad).all()


def test_semi_hard_picks_when_in_band_exists():
    """Construct a batch with exactly one anchor-positive pair and one in-band negative.
    Expect n_triplets >= 1 and loss > 0 (softplus is strictly positive when d_ap > 0)."""
    def from_angle(theta):
        return [math.cos(theta), math.sin(theta)]

    emb = torch.tensor([
        from_angle(0.0),    # anchor, id 0
        from_angle(0.1),    # positive, id 0  -> d_ap ≈ 0.1 after L2-norm
        from_angle(0.25),   # in-band negative, id 1  -> d_an ≈ 0.25 (band = (0.1, 0.4))
        from_angle(0.8),    # out-of-band negative, id 1  -> d_an ≈ 0.78
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    loss, n_triplets = semi_hard_triplet_loss(emb, labels, margin=0.3)
    assert n_triplets >= 1
    assert loss.item() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest training_pipeline/tests/test_loss.py -v -k "softplus or semi_hard"
```

Expected: ImportError (functions don't exist yet).

- [ ] **Step 3: Implement softplus_loss + semi_hard_triplet_loss**

Edit `training_pipeline/src/loss.py`. Keep `_pairwise_dist` and `batch_hard_triplet_loss` (the latter for ablation). Append:

```python
def softplus_loss(d_ap: torch.Tensor, d_an: torch.Tensor) -> torch.Tensor:
    """Soft-margin triplet loss: log(1 + exp(d_ap - d_an))."""
    return F.softplus(d_ap - d_an)


def semi_hard_triplet_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> tuple[torch.Tensor, int]:
    """Semi-hard negative mining triplet loss.

    For each (anchor a, positive p) pair, pick a negative n such that
    d_ap < d_an < d_ap + margin (the "semi-hard" band). Fall back to the
    closest harder-than-positive negative if no in-band one exists.

    Returns (loss, n_triplets). When n_triplets == 0 (degenerate batch) the
    loss is `emb.sum() * 0.0` — a graph-connected zero so `.backward()` is
    safe but produces no gradients. Callers should skip `optimizer.step()`
    in that case.

    Input is L2-normalized internally so all distances live on the unit
    hypersphere — this prevents the model from "satisfying" the loss by
    scaling embedding norms (norm-scale escape).
    """
    emb = F.normalize(emb, p=2, dim=1)
    dist = _pairwise_dist(emb)
    n = labels.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same & ~eye
    neg_mask = ~same

    d_aps: list[torch.Tensor] = []
    d_ans: list[torch.Tensor] = []
    for a in range(n):
        pos_idxs = pos_mask[a].nonzero(as_tuple=False).flatten()
        for p in pos_idxs:
            d_ap = dist[a, p]
            d_an_vec = dist[a]
            # Semi-hard band: d_ap < d_an < d_ap + margin, n must be a true negative.
            in_band = neg_mask[a] & (d_an_vec > d_ap) & (d_an_vec < d_ap + margin)
            in_band_idxs = in_band.nonzero(as_tuple=False).flatten()
            if len(in_band_idxs) > 0:
                # Uniform random pick.
                pick = in_band_idxs[torch.randint(len(in_band_idxs), (1,), device=emb.device).item()]
                d_ans.append(d_an_vec[pick])
                d_aps.append(d_ap)
                continue
            # Fallback: closest harder-than-positive negative.
            harder = neg_mask[a] & (d_an_vec > d_ap)
            harder_idxs = harder.nonzero(as_tuple=False).flatten()
            if len(harder_idxs) > 0:
                pick = harder_idxs[d_an_vec[harder_idxs].argmin()]
                d_ans.append(d_an_vec[pick])
                d_aps.append(d_ap)
                # else: skip this (a, p) — no usable negative.

    if not d_aps:
        return emb.sum() * 0.0, 0

    d_ap_t = torch.stack(d_aps)
    d_an_t = torch.stack(d_ans)
    loss = softplus_loss(d_ap_t, d_an_t).mean()
    return loss, len(d_aps)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest training_pipeline/tests/test_loss.py -v
```

Expected: all 6 new tests pass + the existing `batch_hard_triplet_loss` tests still pass.

- [ ] **Step 5: Commit**

```bash
git add training_pipeline/src/loss.py training_pipeline/tests/test_loss.py
git commit -m "feat(loss): semi-hard triplet mining + soft-margin (L2-norm internal, (loss,n) contract)"
```

---

### Task 6: Eval LFW — LfwPair dataclass, no-drop loader, raw fallback

**Files:**
- Modify: `training_pipeline/src/eval_lfw.py`
- Create: `training_pipeline/tests/test_eval_lfw.py`

- [ ] **Step 1: Write failing tests**

Create `training_pipeline/tests/test_eval_lfw.py`:

```python
"""Tests for the LFW eval loader + fallback + sanity assertions."""
from pathlib import Path

import pytest
from PIL import Image

from training_pipeline.src.eval_lfw import (
    LfwPair,
    _load_image_with_fallback,
    load_pairs_txt,
)


def _make_pairs_txt(p: Path, body: list[str]) -> None:
    p.write_text("\n".join(body) + "\n")


def _make_jpg(path: Path, size: tuple[int, int] = (250, 250)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def test_load_pairs_returns_lfwpair_objects(tmp_path):
    pairs_txt = tmp_path / "pairs.txt"
    # 1 fold × 1 positive × 1 negative.
    _make_pairs_txt(pairs_txt, [
        "1 1",
        "Alice 1 2",
        "Bob 1 Carol 1",
    ])
    pairs = load_pairs_txt(pairs_txt)
    assert len(pairs) == 2
    assert all(isinstance(p, LfwPair) for p in pairs)
    assert pairs[0].name1 == "Alice" and pairs[0].name2 == "Alice" and pairs[0].same == 1
    assert pairs[1].name1 == "Bob" and pairs[1].name2 == "Carol" and pairs[1].same == 0


def test_load_pairs_does_not_drop(tmp_path):
    """Pairs are returned even when their images don't exist on disk."""
    pairs_txt = tmp_path / "pairs.txt"
    _make_pairs_txt(pairs_txt, [
        "1 2",
        "Alice 1 2",
        "Alice 3 4",
        "Bob 1 Carol 1",
        "Dave 1 Eve 1",
    ])
    pairs = load_pairs_txt(pairs_txt)
    assert len(pairs) == 4  # 2 positives + 2 negatives, none dropped


def test_load_image_with_fallback_prefers_aligned(tmp_path):
    aligned = tmp_path / "aligned" / "Alice_0001.jpg"
    raw = tmp_path / "raw" / "Alice_0001.jpg"
    _make_jpg(aligned, size=(160, 160))
    _make_jpg(raw, size=(250, 250))
    tensor = _load_image_with_fallback(aligned, raw)
    assert tensor.shape == (3, 160, 160)


def test_load_image_with_fallback_uses_raw_when_aligned_missing(tmp_path):
    aligned = tmp_path / "aligned" / "Alice_0001.jpg"  # never created
    raw = tmp_path / "raw" / "Alice_0001.jpg"
    _make_jpg(raw, size=(250, 250))
    tensor = _load_image_with_fallback(aligned, raw)
    assert tensor.shape == (3, 160, 160)


def test_load_image_with_fallback_raises_when_both_missing(tmp_path):
    aligned = tmp_path / "aligned" / "Alice_0001.jpg"
    raw = tmp_path / "raw" / "Alice_0001.jpg"
    with pytest.raises((AssertionError, FileNotFoundError)):
        _load_image_with_fallback(aligned, raw)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest training_pipeline/tests/test_eval_lfw.py -v
```

Expected: ImportError on `LfwPair`, `_load_image_with_fallback`.

- [ ] **Step 3: Rewrite eval_lfw.py**

Replace the contents of `training_pipeline/src/eval_lfw.py`:

```python
"""LFW verification: load pairs, embed via fallback-capable image loader,
compute cosine similarity, run 10-fold threshold CV, optionally enforce sanity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .dataset import IMAGENET_MEAN, IMAGENET_STD, eval_transform

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LfwPair:
    name1: str
    idx1: int
    name2: str
    idx2: int
    same: int  # 0 or 1


def load_pairs_txt(pairs_txt: Path) -> list[LfwPair]:
    """Parse LFW pairs.txt and return ALL pairs as LfwPair objects.

    Does NOT filter by disk presence — that's the eval's job, with fallback to raw.
    Format:
        <n_folds> <n_per_fold>
        n_per_fold lines of "name idx1 idx2"      (positives)
        n_per_fold lines of "name1 idx1 name2 idx2" (negatives)
        ... repeated for each fold
    """
    lines = pairs_txt.read_text().splitlines()
    header = lines[0].split()
    n_folds, n_per_fold = int(header[0]), int(header[1])
    pairs: list[LfwPair] = []
    i = 1
    for _ in range(n_folds):
        for _ in range(n_per_fold):
            parts = lines[i].split(); i += 1
            pairs.append(LfwPair(parts[0], int(parts[1]), parts[0], int(parts[2]), 1))
        for _ in range(n_per_fold):
            parts = lines[i].split(); i += 1
            pairs.append(LfwPair(parts[0], int(parts[1]), parts[2], int(parts[3]), 0))
    return pairs


_raw_fallback_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _load_image_with_fallback(aligned: Path, raw: Path) -> torch.Tensor:
    """Load aligned if it exists, otherwise center-crop raw to 80% of min(w,h) and resize.

    Missing raw is a setup bug (not a runtime path); raises AssertionError.
    """
    if aligned.exists():
        return eval_transform()(Image.open(aligned).convert("RGB"))
    assert raw.exists(), f"raw fallback also missing: {raw}"
    img = Image.open(raw).convert("RGB")
    w, h = img.size
    side = int(0.8 * min(w, h))
    l = (w - side) // 2
    t = (h - side) // 2
    img = img.crop((l, t, l + side, t + side)).resize((160, 160), Image.BILINEAR)
    return _raw_fallback_tf(img)


def _pair_paths(pair: LfwPair, aligned_root: Path, raw_root: Path) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    a1 = aligned_root / pair.name1 / f"{pair.name1}_{pair.idx1:04d}.jpg"
    r1 = raw_root / pair.name1 / f"{pair.name1}_{pair.idx1:04d}.jpg"
    a2 = aligned_root / pair.name2 / f"{pair.name2}_{pair.idx2:04d}.jpg"
    r2 = raw_root / pair.name2 / f"{pair.name2}_{pair.idx2:04d}.jpg"
    return (a1, r1), (a2, r2)


class _PairImgDataset(Dataset):
    def __init__(self, unique: list[tuple[Path, Path]]):
        self.unique = unique

    def __len__(self) -> int:
        return len(self.unique)

    def __getitem__(self, i: int):
        aligned, raw = self.unique[i]
        return _load_image_with_fallback(aligned, raw)


def _assert_distribution_sane(metrics: dict) -> None:
    assert metrics["spread"] > 0.05, (
        f"COLLAPSED: spread={metrics['spread']:.4f} "
        f"(pos={metrics['pos_sim_mean']:.3f}, neg={metrics['neg_sim_mean']:.3f})"
    )
    assert metrics["pos_sim_std"] > 0.01, (
        f"COLLAPSED: pos std={metrics['pos_sim_std']:.4f} too tight"
    )
    assert 0.4 < metrics["pos_ratio"] < 0.6, (
        f"LABEL LEAK: {metrics['pos_ratio']:.2%} positive"
    )


def _assert_threshold_sane(metrics: dict) -> None:
    assert 0.0 < metrics["threshold_global"] < 0.9, (
        f"THRESHOLD AT BOUND: {metrics['threshold_global']:.4f}"
    )


@torch.no_grad()
def evaluate_lfw(
    model,
    pairs: list[LfwPair],
    aligned_root: Path,
    raw_root: Path,
    device: str,
    batch_size: int = 128,
    max_pairs: int | None = None,
    strict: bool = True,
) -> dict:
    """Embed all referenced images (with raw fallback), compute cosine sim per pair,
    run 10-fold threshold CV. If `strict`, enforce distribution + threshold sanity.

    The model must implement `embed_normalized(x)` returning unit-norm vectors.
    """
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    # Build unique (aligned, raw) pair list and an index from a hashable key.
    by_key: dict[tuple[str, int], int] = {}
    unique: list[tuple[Path, Path]] = []

    def _key(name: str, idx: int) -> tuple[str, int]:
        return (name, idx)

    for pair in pairs:
        (a1, r1), (a2, r2) = _pair_paths(pair, aligned_root, raw_root)
        if _key(pair.name1, pair.idx1) not in by_key:
            by_key[_key(pair.name1, pair.idx1)] = len(unique)
            unique.append((a1, r1))
        if _key(pair.name2, pair.idx2) not in by_key:
            by_key[_key(pair.name2, pair.idx2)] = len(unique)
            unique.append((a2, r2))

    ds = _PairImgDataset(unique)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    model.eval()
    embs_list: list[torch.Tensor] = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        embs_list.append(model.embed_normalized(x).cpu())
    embs = torch.cat(embs_list, dim=0)

    sims = np.array([
        float((embs[by_key[_key(p.name1, p.idx1)]] * embs[by_key[_key(p.name2, p.idx2)]]).sum())
        for p in pairs
    ])
    labels = np.array([p.same for p in pairs])

    n = len(sims)
    n_folds = 10 if n >= 10 else 1
    fold_size = n // n_folds
    cand = np.linspace(-1, 1, 401)
    accs: list[float] = []
    chosen_thresholds: list[float] = []
    for f in range(n_folds):
        lo, hi = f * fold_size, (f + 1) * fold_size if f < n_folds - 1 else n
        val = np.zeros(n, dtype=bool); val[lo:hi] = True
        train = ~val if n_folds > 1 else val
        best_a, best_t = 0.0, 0.0
        for t in cand:
            a = ((sims[train] > t).astype(int) == labels[train]).mean()
            if a > best_a:
                best_a, best_t = a, float(t)
        accs.append(float(((sims[val] > best_t).astype(int) == labels[val]).mean()))
        chosen_thresholds.append(best_t)

    pos = sims[labels == 1]
    neg = sims[labels == 0]
    metrics = {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "threshold_global": float(np.median(chosen_thresholds)),
        "n_pairs": n,
        "pos_sim_mean": float(pos.mean()) if len(pos) else 0.0,
        "pos_sim_std": float(pos.std()) if len(pos) else 0.0,
        "neg_sim_mean": float(neg.mean()) if len(neg) else 0.0,
        "neg_sim_std": float(neg.std()) if len(neg) else 0.0,
        "spread": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else 0.0,
        "pos_ratio": float(labels.mean()),
    }

    if strict:
        _assert_distribution_sane(metrics)
        _assert_threshold_sane(metrics)

    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest training_pipeline/tests/test_eval_lfw.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add training_pipeline/src/eval_lfw.py training_pipeline/tests/test_eval_lfw.py
git commit -m "feat(eval): LfwPair, no-drop loader, raw fallback, strict/non-strict sanity"
```

---

### Task 7: Strict-mode sanity assertion tests

**Files:**
- Modify: `training_pipeline/tests/test_eval_lfw.py`

- [ ] **Step 1: Write more failing tests**

Append to `training_pipeline/tests/test_eval_lfw.py`:

```python
import pytest

from training_pipeline.src.eval_lfw import (
    _assert_distribution_sane,
    _assert_threshold_sane,
)


def _metrics(pos_mean=0.7, neg_mean=0.3, pos_std=0.05, pos_ratio=0.5, threshold=0.5):
    return {
        "pos_sim_mean": pos_mean,
        "neg_sim_mean": neg_mean,
        "spread": pos_mean - neg_mean,
        "pos_sim_std": pos_std,
        "neg_sim_std": 0.05,
        "pos_ratio": pos_ratio,
        "threshold_global": threshold,
    }


def test_distribution_sane_passes_on_real():
    _assert_distribution_sane(_metrics())  # no exception


def test_distribution_collapse_fires_on_low_spread():
    with pytest.raises(AssertionError, match="COLLAPSED"):
        _assert_distribution_sane(_metrics(pos_mean=0.997, neg_mean=0.996))


def test_distribution_collapse_fires_on_tight_std():
    with pytest.raises(AssertionError, match="too tight"):
        _assert_distribution_sane(_metrics(pos_std=0.001))


def test_distribution_label_leak_fires():
    with pytest.raises(AssertionError, match="LABEL LEAK"):
        _assert_distribution_sane(_metrics(pos_ratio=0.97))


def test_threshold_sane_passes_on_real():
    _assert_threshold_sane(_metrics(threshold=0.45))


def test_threshold_at_lower_bound_fires():
    with pytest.raises(AssertionError, match="THRESHOLD AT BOUND"):
        _assert_threshold_sane(_metrics(threshold=-1.0))


def test_threshold_at_upper_bound_fires():
    with pytest.raises(AssertionError, match="THRESHOLD AT BOUND"):
        _assert_threshold_sane(_metrics(threshold=0.95))
```

- [ ] **Step 2: Run tests to verify they pass** (the functions exist from Task 6)

```bash
pytest training_pipeline/tests/test_eval_lfw.py -v
```

Expected: all 12 tests pass (5 from Task 6 + 7 new).

- [ ] **Step 3: Commit**

```bash
git add training_pipeline/tests/test_eval_lfw.py
git commit -m "test(eval): cover strict-mode sanity assertions (collapse, leak, threshold-at-bound)"
```

---

### Task 8: Update evaluation/eval_lfw.py CLI

**Files:**
- Modify: `evaluation/eval_lfw.py`

- [ ] **Step 1: Read the current CLI**

```bash
cat evaluation/eval_lfw.py
```

- [ ] **Step 2: Rewrite to use the new evaluate_lfw signature**

Replace `evaluation/eval_lfw.py`:

```python
"""LFW 10-fold benchmark CLI. Writes evaluation/results.json with extended metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "process-data"))

from lfw_layout import find_lfw_identity_root  # noqa: E402
from training_pipeline.src.eval_lfw import (  # noqa: E402
    evaluate_lfw,
    load_pairs_txt,
)
from training_pipeline.src.model import FaceEmbedding  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=ROOT / "application/models/best.pt")
    ap.add_argument("--pairs", default=ROOT / "preprocess-data/lfw/pairs.txt")
    ap.add_argument("--aligned-root", default=ROOT / "process-data/lfw_pairs")
    ap.add_argument("--raw-root", default=None,
                    help="Override raw LFW root; default = auto-discover under preprocess-data/lfw")
    ap.add_argument("--out", default=ROOT / "evaluation/results.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    raw_root = Path(args.raw_root) if args.raw_root else find_lfw_identity_root(ROOT / "preprocess-data/lfw")

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = FaceEmbedding(embedding_dim=ckpt["cfg"]["train"]["embedding_dim"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    pairs = load_pairs_txt(Path(args.pairs))
    metrics = evaluate_lfw(
        model, pairs,
        aligned_root=Path(args.aligned_root),
        raw_root=raw_root,
        device=args.device,
        strict=True,
    )

    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the CLI imports cleanly**

```bash
python -c "import evaluation.eval_lfw"
```

Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add evaluation/eval_lfw.py
git commit -m "feat(eval-cli): wire new LfwPair API, auto-discover raw root, strict mode"
```

---

### Task 9: Create evaluation/eval_pins.py

**Files:**
- Create: `evaluation/eval_pins.py`

- [ ] **Step 1: Implement the Pins evaluator**

```python
"""Pins Face Recognition cross-dataset overfit check.

Pins identities are disjoint from CASIA (training) AND LFW (eval). This is the
"is the model real" test. Uses the LFW-tuned threshold as a FIXED parameter —
never re-tunes on Pins (that would just measure 'can we fit Pins?').
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training_pipeline.src.dataset import eval_transform  # noqa: E402
from training_pipeline.src.eval_lfw import _assert_distribution_sane  # noqa: E402
from training_pipeline.src.model import FaceEmbedding  # noqa: E402

N_POS = 1500
N_NEG = 1500
SEED = 0


def _build_pairs(pins_root: Path) -> list[tuple[Path, Path, int]]:
    rng = random.Random(SEED)
    celebs = sorted(d for d in pins_root.iterdir() if d.is_dir())
    by_celeb = {c.name: sorted(c.glob("*.jpg")) for c in celebs}
    by_celeb = {k: v for k, v in by_celeb.items() if len(v) >= 2}
    names = list(by_celeb.keys())

    pairs: list[tuple[Path, Path, int]] = []
    for _ in range(N_POS):
        n = rng.choice(names)
        a, b = rng.sample(by_celeb[n], 2)
        pairs.append((a, b, 1))
    for _ in range(N_NEG):
        n1, n2 = rng.sample(names, 2)
        a = rng.choice(by_celeb[n1])
        b = rng.choice(by_celeb[n2])
        pairs.append((a, b, 0))
    rng.shuffle(pairs)
    return pairs


class _PinsDataset(Dataset):
    def __init__(self, paths: list[Path], mtcnn: MTCNN):
        self.paths = paths
        self.mtcnn = mtcnn
        self.tf = eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        face = self.mtcnn(img)
        if face is None:
            # Fallback: center-crop.
            w, h = img.size
            side = int(0.8 * min(w, h))
            l = (w - side) // 2
            t = (h - side) // 2
            return self.tf(img.crop((l, t, l + side, t + side)).resize((160, 160)))
        return self.tf(Image.fromarray(face.byte().permute(1, 2, 0).cpu().numpy()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins-root",
                    default=ROOT / "preprocess-data/pins/105_classes_pins_dataset")
    ap.add_argument("--checkpoint", default=ROOT / "application/models/best.pt")
    ap.add_argument("--lfw-results", default=ROOT / "evaluation/results.json")
    ap.add_argument("--out", default=ROOT / "evaluation/results_pins.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    lfw = json.loads(Path(args.lfw_results).read_text())
    threshold = float(lfw["threshold_global"])
    print(f"using LFW threshold: {threshold:.4f}")

    pairs = _build_pairs(Path(args.pins_root))
    unique = sorted({p for a, b, _ in pairs for p in (a, b)})
    print(f"unique images: {len(unique)}  pairs: {len(pairs)}")

    mtcnn = MTCNN(image_size=160, margin=0, post_process=False, device=args.device, keep_all=False)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = FaceEmbedding(embedding_dim=ckpt["cfg"]["train"]["embedding_dim"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    ds = _PinsDataset(unique, mtcnn)
    dl = DataLoader(ds, batch_size=64, num_workers=0)  # MTCNN doesn't play well with DataLoader workers.

    embs: list[torch.Tensor] = []
    for x in dl:
        x = x.to(args.device, non_blocking=True)
        with torch.no_grad():
            embs.append(model.embed_normalized(x).cpu())
    embs_t = torch.cat(embs, dim=0)
    path_to_idx = {p: i for i, p in enumerate(unique)}

    sims = np.array([float((embs_t[path_to_idx[a]] * embs_t[path_to_idx[b]]).sum())
                     for a, b, _ in pairs])
    labels = np.array([l for _, _, l in pairs])

    pos = sims[labels == 1]
    neg = sims[labels == 0]
    metrics = {
        "dataset": "pins-face-recognition (105 celebs, disjoint from CASIA + LFW)",
        "n_pairs": int(len(sims)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "pos_sim_mean": float(pos.mean()),
        "pos_sim_std": float(pos.std()),
        "neg_sim_mean": float(neg.mean()),
        "neg_sim_std": float(neg.std()),
        "spread": float(pos.mean() - neg.mean()),
        "pos_ratio": float(labels.mean()),
        "lfw_threshold_used": threshold,
        "accuracy_at_lfw_threshold": float(((sims > threshold).astype(int) == labels).mean()),
    }
    # Reference-only: best-on-pins threshold (NOT the headline number).
    cand = np.linspace(-1, 1, 401)
    accs = np.array([((sims > t).astype(int) == labels).mean() for t in cand])
    metrics["accuracy_at_pins_tuned_threshold"] = float(accs.max())
    metrics["pins_tuned_threshold"] = float(cand[int(accs.argmax())])

    _assert_distribution_sane(metrics)  # distribution-only; no threshold-bound check.

    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

```bash
python -c "import evaluation.eval_pins"
```

Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add evaluation/eval_pins.py
git commit -m "feat(eval): Pins cross-dataset evaluator using fixed LFW threshold"
```

---

### Task 10: Two-phase train.py

**Files:**
- Modify: `training_pipeline/src/train.py`

- [ ] **Step 1: Replace `training_pipeline/src/train.py`**

```python
"""Two-phase Siamese training: softmax warmup (CE) → semi-hard triplet."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import FaceDataset, PKSampler, train_transform
from .eval_lfw import LfwPair, evaluate_lfw, load_pairs_txt
from .loss import semi_hard_triplet_loss
from .model import ClassifierHead, FaceEmbedding
from .utils import AverageMeter, set_seed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "process-data"))
from lfw_layout import find_lfw_identity_root  # noqa: E402


def _build_optimizer(params_backbone, params_head, lr_backbone, lr_head, wd):
    return torch.optim.AdamW(
        [
            {"params": params_backbone, "lr": lr_backbone},
            {"params": params_head,     "lr": lr_head},
        ],
        weight_decay=wd,
    )


def _split_params(model: FaceEmbedding):
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if "backbone.fc" in name else backbone).append(p)
    return backbone, head


def _cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def _run_lfw_probe(model, pairs, aligned_root, raw_root, device, max_pairs):
    """Non-strict in-loop probe. Returns metrics dict; never aborts training."""
    return evaluate_lfw(
        model, pairs[:max_pairs] if max_pairs else pairs,
        aligned_root=aligned_root, raw_root=raw_root,
        device=device, strict=False,
    )


def run_training(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(cfg["checkpoints_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(cfg["tensorboard_dir"]);   tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    train_ds = FaceDataset(Path(cfg["manifest"]), split="train", transform=train_transform())
    n_identities = len(set(train_ds.labels))

    # LFW probe inputs (shared across both phases).
    pairs = load_pairs_txt(Path(cfg["eval"]["pairs_txt"]))
    aligned_root = ROOT / "process-data/lfw_pairs"
    raw_root = find_lfw_identity_root(ROOT / "preprocess-data/lfw")

    model = FaceEmbedding(embedding_dim=cfg["train"]["embedding_dim"]).to(device)
    classifier = ClassifierHead(cfg["train"]["embedding_dim"], n_identities).to(device)

    history = {"train_loss": [], "lfw_acc": [], "spread": [], "phase": []}
    best_acc = 0.0

    # ---------- Phase 1: softmax warmup ----------
    phase1_epochs = cfg["train"]["phase1_epochs"]
    phase1_batches = cfg["train"]["batches_per_epoch"]
    phase1_bs = cfg["train"]["phase1_batch_size"]

    p1_sampler = RandomSampler(
        train_ds, replacement=True,
        num_samples=phase1_batches * phase1_bs,
    )
    p1_loader = DataLoader(
        train_ds, sampler=p1_sampler,
        batch_size=phase1_bs,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg["train"]["num_workers"] > 0),
    )

    backbone_params, head_params = _split_params(model)
    p1_optim = _build_optimizer(
        backbone_params, list(classifier.parameters()) + head_params,
        cfg["train"]["lr_backbone"], cfg["train"]["lr_head"], cfg["train"]["weight_decay"],
    )
    ce = torch.nn.CrossEntropyLoss()
    p1_total_steps = phase1_epochs * phase1_batches
    p1_warmup = cfg["train"]["warmup_steps"]
    step = 0

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    for epoch in range(1, phase1_epochs + 1):
        model.train(); classifier.train()
        meter = AverageMeter()
        pbar = tqdm(p1_loader, desc=f"P1 epoch {epoch}/{phase1_epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            for pg, base in zip(p1_optim.param_groups, [cfg["train"]["lr_backbone"], cfg["train"]["lr_head"]]):
                pg["lr"] = _cosine_lr(step, p1_total_steps, p1_warmup, base)
            p1_optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                emb = model(imgs)
                logits = classifier(emb)
                loss = ce(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(p1_optim)
            scaler.update()
            meter.update(loss.item(), imgs.size(0))
            pbar.set_postfix(loss=meter.avg)
            writer.add_scalar("p1/train_loss_step", loss.item(), step)
            step += 1
        # End of P1 epoch: LFW probe.
        metrics = _run_lfw_probe(model, pairs, aligned_root, raw_root, device,
                                 cfg["eval"]["max_pairs_inloop"])
        history["train_loss"].append(meter.avg)
        history["lfw_acc"].append(metrics["mean_acc"])
        history["spread"].append(metrics["spread"])
        history["phase"].append(1)
        print(f"[P1 epoch {epoch}] train_loss={meter.avg:.4f}  lfw_acc={metrics['mean_acc']:.4f}  "
              f"spread={metrics['spread']:.4f}  pos={metrics['pos_sim_mean']:.3f}  neg={metrics['neg_sim_mean']:.3f}")
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))

    # P1 end-of-phase gate.
    if history["spread"][-1] <= 0.05:
        raise RuntimeError(
            f"phase 1 produced collapsed embeddings (spread={history['spread'][-1]:.4f}). "
            f"Check data, augmentations, learning rate. Aborting before phase 2."
        )

    torch.save(
        {"model": model.state_dict(), "classifier": classifier.state_dict(), "cfg": cfg},
        ckpt_dir / "phase1_end.pt",
    )

    # ---------- Phase 2: semi-hard triplet ----------
    del classifier  # explicit; the phase-2 optimizer won't see it.
    phase2_epochs = cfg["train"]["phase2_epochs"]
    phase2_batches = cfg["train"]["batches_per_epoch"]

    p2_sampler = PKSampler(
        train_ds.labels,
        p=cfg["train"]["p"],
        k=cfg["train"]["k"],
        num_batches=phase2_batches,
        seed=cfg["seed"],
    )
    p2_loader = DataLoader(
        train_ds, batch_sampler=p2_sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg["train"]["num_workers"] > 0),
    )

    backbone_params, head_params = _split_params(model)
    p2_optim = _build_optimizer(
        backbone_params, head_params,
        cfg["train"]["lr_backbone"], cfg["train"]["lr_head"], cfg["train"]["weight_decay"],
    )
    p2_total_steps = phase2_epochs * phase2_batches
    p2_warmup = cfg["train"]["warmup_steps"]
    step = 0
    zero_triplet_streak = 0

    for epoch in range(1, phase2_epochs + 1):
        model.train()
        meter = AverageMeter()
        n_triplet_meter = AverageMeter()
        pbar = tqdm(p2_loader, desc=f"P2 epoch {epoch}/{phase2_epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            for pg, base in zip(p2_optim.param_groups, [cfg["train"]["lr_backbone"], cfg["train"]["lr_head"]]):
                pg["lr"] = _cosine_lr(step, p2_total_steps, p2_warmup, base)
            p2_optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                emb = model(imgs)
                loss, n_triplets = semi_hard_triplet_loss(emb, labels, margin=cfg["train"]["margin"])
            if n_triplets > 0:
                scaler.scale(loss).backward()
                scaler.step(p2_optim)
                scaler.update()
                zero_triplet_streak = 0
            else:
                zero_triplet_streak += 1
                if zero_triplet_streak >= 100:
                    raise RuntimeError(
                        "100 consecutive batches produced zero usable triplets — sampler/mining bug. "
                        "Inspect PKSampler output and semi_hard_triplet_loss selection."
                    )
            meter.update(loss.item(), imgs.size(0))
            n_triplet_meter.update(n_triplets, 1)
            pbar.set_postfix(loss=meter.avg, n_tri=n_triplet_meter.avg)
            writer.add_scalar("p2/train_loss_step", loss.item(), step)
            writer.add_scalar("p2/n_triplets_step", n_triplets, step)
            step += 1
        # End of P2 epoch: LFW probe.
        metrics = _run_lfw_probe(model, pairs, aligned_root, raw_root, device,
                                 cfg["eval"]["max_pairs_inloop"])
        history["train_loss"].append(meter.avg)
        history["lfw_acc"].append(metrics["mean_acc"])
        history["spread"].append(metrics["spread"])
        history["phase"].append(2)
        print(f"[P2 epoch {epoch}] train_loss={meter.avg:.4f}  lfw_acc={metrics['mean_acc']:.4f}  "
              f"spread={metrics['spread']:.4f}  n_triplets/avg={n_triplet_meter.avg:.1f}")
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))

        # Update best.pt ONLY if spread > 0.05 (don't promote a collapsed model).
        if metrics["mean_acc"] > best_acc and metrics["spread"] > 0.05:
            best_acc = metrics["mean_acc"]
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir / "best.pt")

    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir / "last.pt")
    return {"best_acc": best_acc, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path(__file__).resolve().parent.parent / "configs/train.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    result = run_training(cfg)
    print(f"best LFW probe acc: {result['best_acc']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

```bash
python -c "from training_pipeline.src.train import run_training; print('ok')"
```

Expected: prints `ok`. (Won't actually train — just verifying the file is importable.)

- [ ] **Step 3: Commit**

```bash
git add training_pipeline/src/train.py
git commit -m "feat(train): two-phase training — softmax warmup -> semi-hard triplet, phase-1 collapse gate, no-promote-collapsed best.pt"
```

---

### Task 11: Update training config

**Files:**
- Modify: `training_pipeline/configs/train.yaml`

- [ ] **Step 1: Replace train.yaml**

```yaml
manifest: process-data/manifest.parquet
checkpoints_dir: training_pipeline/checkpoints
tensorboard_dir: training_pipeline/tensorboard_logs
seed: 42

train:
  phase1_epochs: 3
  phase2_epochs: 17
  batches_per_epoch: 1500
  phase1_batch_size: 128
  p: 32
  k: 4
  margin: 0.3
  lr_head: 5.0e-4
  lr_backbone: 1.0e-4
  weight_decay: 5.0e-4
  warmup_steps: 500
  num_workers: 8
  embedding_dim: 512

eval:
  pairs_txt: preprocess-data/lfw/pairs.txt
  max_pairs_inloop: 1000
  batch_size: 128
```

- [ ] **Step 2: Commit**

```bash
git add training_pipeline/configs/train.yaml
git commit -m "feat(config): phase split + raised LRs + canonical FaceNet/MassFace hyperparams"
```

---

### Task 12: Update smoke test for two-phase

**Files:**
- Modify: `training_pipeline/tests/test_smoke.py`

- [ ] **Step 1: Read current smoke test**

```bash
cat training_pipeline/tests/test_smoke.py
```

- [ ] **Step 2: Adapt for two-phase**

Replace `training_pipeline/tests/test_smoke.py`:

```python
"""End-to-end smoke test: a tiny two-phase run on synthetic data must not collapse."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from training_pipeline.src.dataset import IMAGENET_MEAN, IMAGENET_STD
from training_pipeline.src.train import run_training


@pytest.fixture
def synthetic_manifest(tmp_path: Path):
    """Build a tiny manifest with 5 identities × 4 images = 20 samples.

    Each identity gets a distinct color so the model has a learnable signal.
    """
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


@pytest.mark.slow
def test_two_phase_smoke_no_collapse(synthetic_manifest, monkeypatch, tmp_path):
    """Run 1 phase-1 epoch + 1 phase-2 epoch on synthetic data. Spread must end > 0.05."""
    workdir, manifest = synthetic_manifest

    # FaceDataset reads `ROOT / path` — make ROOT point at the tmp workdir.
    import training_pipeline.src.dataset as ds_mod
    monkeypatch.setattr(ds_mod, "ROOT", workdir)

    # Stub LFW probe — we don't have LFW data in the smoke test.
    import training_pipeline.src.train as train_mod
    def _stub_probe(*args, **kwargs):
        return {"mean_acc": 0.5, "spread": 0.10, "pos_sim_mean": 0.6, "neg_sim_mean": 0.5}
    monkeypatch.setattr(train_mod, "_run_lfw_probe", _stub_probe)

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

    # Override FaceEmbedding to use a tiny CNN to keep the smoke test fast.
    from training_pipeline.src import model as model_mod
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyEmbed(nn.Module):
        def __init__(self, embedding_dim: int = 32, pretrained: bool = False):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
            self.fc = nn.Linear(16 * 80 * 80, embedding_dim)
            self.backbone = type("X", (), {"fc": self.fc})()  # so _split_params finds 'backbone.fc'
        def forward(self, x):
            h = F.relu(self.conv(x)).flatten(1)
            return self.fc(h)
        def embed_normalized(self, x):
            return F.normalize(self.forward(x), p=2, dim=1)

    monkeypatch.setattr(model_mod, "FaceEmbedding", TinyEmbed)
    monkeypatch.setattr(train_mod, "FaceEmbedding", TinyEmbed)

    result = run_training(cfg)
    assert "history" in result
    # Smoke test just needs to NOT crash + NOT fail the phase-1 gate (stubbed spread > 0.05).
    assert (tmp_path / "ckpts" / "phase1_end.pt").exists()
    assert (tmp_path / "ckpts" / "last.pt").exists()
```

- [ ] **Step 3: Run the smoke test**

```bash
pytest training_pipeline/tests/test_smoke.py -v -m slow
```

Expected: passes. If it hits OOM / time issues, reduce embedding_dim or batches_per_epoch further.

- [ ] **Step 4: Commit**

```bash
git add training_pipeline/tests/test_smoke.py
git commit -m "test(smoke): two-phase end-to-end on synthetic data with stubbed LFW probe"
```

---

### Task 13: Local test gate before VM

**Files:** none

- [ ] **Step 1: Run all tests**

```bash
pytest -v 2>&1 | tail -40
```

Expected: all green except integration tests that need an LFW sample / checkpoint (those should `SKIPPED`).

- [ ] **Step 2: If anything red, fix it before pushing to VM. No commit needed unless fixing.**

---

### Task 14: Push to VM and run training

**Files:** none (operational)

- [ ] **Step 1: Sync code to VM**

```bash
SSHPASS='Dickenson@1234#' sshpass -e rsync -avz -e "ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no" \
  --exclude '.venv' --exclude '__pycache__' --exclude 'preprocess-data' --exclude 'process-data/train' --exclude 'process-data/val' --exclude 'process-data/lfw_pairs' \
  --exclude 'training_pipeline/checkpoints' --exclude 'application/models' --exclude 'evaluation/figs' \
  ./ root@124.197.18.72:/root/IT4432E_Project/
```

- [ ] **Step 2: Kick off training in tmux**

```bash
SSHPASS='Dickenson@1234#' sshpass -e ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@124.197.18.72 '
  cd /root/IT4432E_Project && source .venv/bin/activate &&
  tmux kill-session -t train 2>/dev/null;
  tmux new-session -d -s train "cd /root/IT4432E_Project && source .venv/bin/activate && python -m training_pipeline.src.train --config training_pipeline/configs/train.yaml 2>&1 | tee /root/IT4432E_Project/training_pipeline/checkpoints/train.log" &&
  echo "training started in tmux session train"
'
```

- [ ] **Step 3: Monitor**

```bash
SSHPASS='Dickenson@1234#' sshpass -e ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@124.197.18.72 '
  tail -30 /root/IT4432E_Project/training_pipeline/checkpoints/train.log
'
```

Repeat every couple minutes. Expected: P1 loss decreases; end of P1 prints `spread > 0.05`. P2 loss decreases, lfw_acc rises. ~20 min total.

- [ ] **Step 4: If P1 gate fires** (`spread <= 0.05`): training aborts with a clear error. Inspect the log, fix the root cause (most likely: dataset / augmentations / LRs), re-sync, restart. Do not advance to Task 15 until you have a phase-1 spread > 0.05.

---

### Task 15: Pull artifacts and run final evals

**Files:** receives `application/models/best.pt`, `evaluation/results.json`, `evaluation/results_pins.json`, `training_pipeline/checkpoints/history.json`, `training_pipeline/checkpoints/train.log`

- [ ] **Step 1: Pull checkpoint + history**

```bash
SSHPASS='Dickenson@1234#' sshpass -e scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  root@124.197.18.72:/root/IT4432E_Project/training_pipeline/checkpoints/best.pt \
  ./application/models/best.pt

SSHPASS='Dickenson@1234#' sshpass -e scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  root@124.197.18.72:/root/IT4432E_Project/training_pipeline/checkpoints/history.json \
  ./training_pipeline/checkpoints/history.json
```

- [ ] **Step 2: Run LFW eval on VM** (where data lives)

```bash
SSHPASS='Dickenson@1234#' sshpass -e ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@124.197.18.72 '
  cd /root/IT4432E_Project && source .venv/bin/activate && python -m evaluation.eval_lfw 2>&1 | tail -20
'
```

Expected: a `results.json` blob with mean_acc > 0.80, spread > 0.30, threshold ∈ (0.0, 0.9), `pos_ratio` close to 0.5.

- [ ] **Step 3: Run Pins cross-dataset eval on VM**

```bash
SSHPASS='Dickenson@1234#' sshpass -e ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@124.197.18.72 '
  cd /root/IT4432E_Project && source .venv/bin/activate && python -m evaluation.eval_pins 2>&1 | tail -25
'
```

Expected: `accuracy_at_lfw_threshold > 0.70`, spread > 0.20.

- [ ] **Step 4: Pull both result JSONs back**

```bash
SSHPASS='Dickenson@1234#' sshpass -e scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  root@124.197.18.72:/root/IT4432E_Project/evaluation/results.json \
  root@124.197.18.72:/root/IT4432E_Project/evaluation/results_pins.json \
  ./evaluation/
```

- [ ] **Step 5: Spin up the app locally and manually verify**

```bash
.venv/bin/python -m uvicorn application.backend.main:app --reload &
APP_PID=$!
# Open http://localhost:8000 in a browser. Enroll yourself; verify; enroll a different person (or use a photo); verify both.
# Same face → match. Different face → reject.
# When done:
kill $APP_PID
```

The diagnostic script from the original collapse hunt is a fast headline check:

```bash
SSHPASS='Dickenson@1234#' sshpass -e ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@124.197.18.72 'cd /root/IT4432E_Project && source .venv/bin/activate && python /tmp/diagnose.py 2>&1 | tail -20'
```

Expected: same-identity cosine sim ≫ different-identity cosine sim (spread > 0.3).

- [ ] **Step 6: Commit artifacts**

```bash
git add application/models/best.pt training_pipeline/checkpoints/history.json evaluation/results.json evaluation/results_pins.json
git commit -m "feat(model): trained checkpoint + LFW/Pins eval results (real discrimination this time)"
```

---

### Task 16: Update notebooks + README

**Files:**
- Re-execute on VM: `process-data/data_processing.ipynb`, `training_pipeline/training_results.ipynb`, `evaluation/evaluation_results.ipynb`
- Modify: `README.md`

- [ ] **Step 1: Re-execute notebooks on VM**

```bash
SSHPASS='Dickenson@1234#' sshpass -e ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@124.197.18.72 '
  cd /root/IT4432E_Project && source .venv/bin/activate &&
  cd process-data && jupyter nbconvert --to notebook --execute --inplace data_processing.ipynb &&
  cd ../training_pipeline && jupyter nbconvert --to notebook --execute --inplace training_results.ipynb &&
  cd ../evaluation && jupyter nbconvert --to notebook --execute --inplace evaluation_results.ipynb
'
```

- [ ] **Step 2: Pull executed notebooks + figures back**

```bash
SSHPASS='Dickenson@1234#' sshpass -e scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  root@124.197.18.72:/root/IT4432E_Project/process-data/data_processing.ipynb ./process-data/

SSHPASS='Dickenson@1234#' sshpass -e scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  root@124.197.18.72:/root/IT4432E_Project/training_pipeline/training_results.ipynb ./training_pipeline/

SSHPASS='Dickenson@1234#' sshpass -e scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  root@124.197.18.72:/root/IT4432E_Project/evaluation/evaluation_results.ipynb ./evaluation/

SSHPASS='Dickenson@1234#' sshpass -e scp -r -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  'root@124.197.18.72:/root/IT4432E_Project/evaluation/figs/*' ./evaluation/figs/
```

- [ ] **Step 3: Rewrite README result paragraph**

Replace the headline number block in `README.md` (the paragraph that currently claims 97.48%) with real numbers from `evaluation/results.json` and `evaluation/results_pins.json`. Show: LFW mean_acc ± std, threshold, spread; Pins accuracy_at_lfw_threshold, spread. Add a short "First attempt collapsed; redesigned per `docs/superpowers/specs/2026-05-18-siamese-training-fix-design.md`" sentence under it.

Concrete text (fill in the numbers from the real results.json):

```markdown
LFW verification accuracy: **<MEAN_ACC>% ± <STD>%** (10-fold CV, 6000 pairs, threshold <THRESHOLD>, pos/neg cosine spread <SPREAD>). Training ran ~<MINUTES> minutes on a single H100 80GB.

Cross-dataset overfit check: **<PINS_ACC>%** on 3000 Pins pairs (105 celebrities, disjoint from CASIA and LFW) at the LFW-tuned threshold. Confirms the model generalizes; not LFW-specific overfit.

> First attempt collapsed silently (all embeddings ≈ same vector; LFW eval looked like 97.48% but was label-leak from dropped pairs). Redesigned per `docs/superpowers/specs/2026-05-18-siamese-training-fix-design.md` — softmax warmup → semi-hard triplet, L2-norm in loss, raw-fallback eval with strict sanity checks.
```

- [ ] **Step 4: Commit everything**

```bash
git add process-data/data_processing.ipynb training_pipeline/training_results.ipynb evaluation/evaluation_results.ipynb evaluation/figs/ README.md
git commit -m "docs(results): re-executed notebooks + figures + honest README with real LFW+Pins numbers"
git push
```

---

## Verification checklist (after Task 16)

- [ ] `pytest -v` passes locally (slow tests can stay marked `slow`).
- [ ] `evaluation/results.json` has `mean_acc > 0.80`, `spread > 0.3`, `0.0 < threshold_global < 0.9`, `0.4 < pos_ratio < 0.6`.
- [ ] `evaluation/results_pins.json` has `accuracy_at_lfw_threshold > 0.70`, `spread > 0.2`.
- [ ] Browser test: enrolling person A and verifying person A returns match; verifying person B returns no-match.
- [ ] `application/models/best.pt` loaded by app cleanly; `/verify` returns sensible cosine values (not all 0.997).
- [ ] H100 VM shut down via provider dashboard.
