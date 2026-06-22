# Face-Recognition Model Consolidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate three face-recognition models (ArcFace, AdaFace, FaceNet) onto `main` under one organized tree, give the app a runtime model selector, and standardize evaluation so every model reports Accuracy/Precision/Recall/F1/ROC-AUC from one shared evaluator.

**Architecture:** Side-by-side organization — each model keeps its own training code under `models/<name>/`; common pipeline stages move to `shared/`. A model **registry** holds every per-model difference (backbone builder, checkpoint format, input size, normalization, threshold, TTA). The app loads all available checkpoints, routes enroll/verify per selected model with its own embedding store, and a shared evaluator embeds pairs at each model's contract to emit a canonical metric schema.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.6, torchvision, facenet-pytorch (MTCNN + InceptionResnetV1), FastAPI, numpy, scikit-learn, pytest.

## Global Constraints

- `requires-python = ">=3.10"`; torch ≥2.6, torchvision ≥0.21, facenet-pytorch ≥2.6, numpy ≥2.0, scikit-learn ≥1.5, fastapi ≥0.115 (copy version floors verbatim from existing `pyproject.toml`).
- Per-model inference contracts **must not be unified** (a single global normalization silently regresses ArcFace):
  | Model | Input size | Normalization (mean / std) | Default threshold | TTA |
  |---|---|---|---|---|
  | arcface | 160 | ImageNet `[.485,.456,.406]` / `[.229,.224,.225]` | 0.565 | flag, default off |
  | adaface | 112 | `[.5,.5,.5]` / `[.5,.5,.5]` | 0.5 | n/a |
  | facenet | 160 | `[.5,.5,.5]` / `[.5,.5,.5]` | 0.5 (calibration knob — tune from eval) | n/a |
- All embedders emit **L2-normalized 512-d** vectors. Embeddings are not comparable across models → per-model stores.
- Preserve the `embedding_version` enrollment guard for **every** model (checkpoint hash + model name + TTA flag).
- Weights are **gitignored**; never commit `.pt`/`.pth`. No Git LFS.
- Canonical eval schema (exact keys): `{accuracy, precision, recall, f1, roc_auc, threshold, std_acc, tp, fp, tn, fn}`.
- Confusion metrics computed at the **median 10-fold CV threshold**.
- Ruff line-length 100. Frequent commits, TDD, DRY, YAGNI.
- App is unauthenticated (local demo) — documented, not fixed.

**Source branches to pull from:** `origin/feat/arcface-tta` (fullest base, has the version guard), `origin/thanh/feat/adaface` (IResNet50 + AdaFace eval with full metrics), `origin/Facenet-Model` (`casia.ipynb` + `facenet_best.pth` on disk at `/Users/n3m0/Downloads/`).

---

## File Structure

```
models/
  arcface/__init__.py, model.py, arcface_head.py, train.py, configs/, src helpers
  adaface/__init__.py, iresnet.py        (inference + eval only — no trainer exists in repo)
  facenet/__init__.py, casia.ipynb, facenet_model.py   (notebook is the trainer)
application/backend/
  registry.py      (NEW) model name → ModelSpec + builders
  embed_wrapper.py (NEW) backbone → embed_normalized / embed_tta
  inference.py     (MODIFY) Embedder dispatches via registry
  face_align.py    (MODIFY) aligner parameterized by size + norm
  db.py            (KEEP)   per-model store + version guard (already exists)
  main.py          (MODIFY) /models endpoint + model field routing
application/frontend/
  index.html, app.js (MODIFY) model dropdown
evaluation/
  metrics.py       (NEW) pure compute_metrics
  evaluate.py      (NEW) per-model LFW eval CLI
  compare.py       (NEW) aggregate results → comparison.{md,csv}
  results/         per-model JSON + comparison table
shared/
  preprocess-data/, process-data/, infra/
pyproject.toml     (MODIFY) package includes
.gitignore, README.md (MODIFY)
```

---

### Task 1: Bootstrap consolidation branch + folder skeleton

Create the branch off the fullest base and reorganize into `models/` + `shared/`. This is mechanical file movement (verbatim, no rewriting) plus package config. The deliverable is a tree where `pytest` collects with the new import paths.

**Files:**
- Branch from `origin/feat/arcface-tta`
- Move: `training_pipeline/src/*` → `models/arcface/`, `preprocess-data|process-data|infra` → `shared/`
- Copy from `origin/thanh/feat/adaface`: `application/backend/iresnet.py` → `models/adaface/iresnet.py` (inference/eval backbone only — **no AdaFace trainer exists in the repo**, see Step 4 note), its `adaface_evaluation.ipynb` → `evaluation/`
- Copy from `origin/Facenet-Model`: `casia.ipynb` → `models/facenet/casia.ipynb`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the consolidation branch off the fullest base**

```bash
cd /home/n3m0/Code/IT4432E_Project
git fetch origin
git checkout -b feat/model-consolidation origin/feat/arcface-tta
```

- [ ] **Step 1b: Ensure the project + dev deps are installed**

The unit tests import `torch`, `torchvision`, and `facenet_pytorch` — these are
normal project dependencies (see `pyproject.toml`), not optional. The execution
environment must have them. Install editable with dev extras (one-time):

```bash
python -m pip install -e ".[dev]"
python -c "import torch, torchvision, facenet_pytorch; print('deps ok')"
```

Expected: `deps ok`. If this box genuinely cannot install torch, run the plan
where it can — the unit tests are torch-dependent by nature. (No GPU and no model
weights are required; CPU torch is enough.)

- [ ] **Step 2: Move common pipeline stages into `shared/`**

```bash
git mv preprocess-data shared/preprocess-data
git mv process-data    shared/process-data
git mv infra           shared/infra
mkdir -p models/arcface models/adaface models/facenet
touch models/__init__.py models/arcface/__init__.py models/adaface/__init__.py models/facenet/__init__.py
```

- [ ] **Step 3: Move ArcFace training code into `models/arcface/`**

```bash
git mv training_pipeline/src/model.py        models/arcface/model.py
git mv training_pipeline/src/arcface_head.py models/arcface/arcface_head.py
git mv training_pipeline/src/loss.py         models/arcface/loss.py
git mv training_pipeline/src/dataset.py      models/arcface/dataset.py
git mv training_pipeline/src/train.py        models/arcface/train.py
git mv training_pipeline/src/utils.py        models/arcface/utils.py
git mv training_pipeline/src/eval_lfw.py     models/arcface/eval_lfw.py
git mv training_pipeline/configs            models/arcface/configs
git mv training_pipeline/tests              models/arcface/tests
git rm -r training_pipeline/checkpoints --ignore-unmatch
git rm training_pipeline/__init__.py training_pipeline/src/__init__.py --ignore-unmatch
git rm training_pipeline/training_results.ipynb --ignore-unmatch
```

- [ ] **Step 4: Bring in AdaFace backbone + FaceNet notebook**

The AdaFace model was trained by an external `train_local.py` that was **never
committed** to the branch — `training_pipeline/src/train.py` on the adaface branch
is still the Siamese trainer (imports `.dataset/.loss/.model`, not `iresnet`).
So copy **only** the AdaFace inference backbone + its eval notebook; do **not**
copy a trainer for it. Training AdaFace is out of scope (no retraining anywhere).

```bash
git show origin/thanh/feat/adaface:application/backend/iresnet.py > models/adaface/iresnet.py
git show origin/Facenet-Model:casia.ipynb > models/facenet/casia.ipynb
git show origin/thanh/feat/adaface:evaluation/adaface_evaluation.ipynb > evaluation/adaface_evaluation.ipynb
```

A `models/adaface/README.md` one-liner records where the weights come from:
```bash
printf '# AdaFace\n\nIResNet50 + AdaFace, trained externally (train_local.py, not in repo).\nDrop the checkpoint at `application/models/adaface.pt`. Inference backbone: `iresnet.py`.\nEval: `evaluation/adaface_evaluation.ipynb` (8-suite) or `python -m evaluation.evaluate --model adaface`.\n' > models/adaface/README.md
```

- [ ] **Step 5: Rewrite the moved import paths**

Old ArcFace code imported `from training_pipeline.src.model import FaceEmbedding`.
The ArcFace package is now `models.arcface`. **Scope the rewrite to the ArcFace
code + the app + eval only** — do NOT rewrite `models/adaface/` or
`models/facenet/`, whose `training_pipeline.src` references mean their *own*
pipelines and which are reference-only (no retraining happens during
consolidation):

```bash
# Only arcface-moved code, the app, and evaluation get the arcface rewrite.
grep -rl "training_pipeline.src" --include=*.py models/arcface application evaluation | tee /tmp/imports.txt
xargs sed -i 's/training_pipeline\.src\./models.arcface./g' < /tmp/imports.txt
# process-data is now shared/process-data; fix sys.path inserts and lfw_layout imports
grep -rl "process-data" --include=*.py models/arcface application evaluation \
  | xargs -r sed -i 's#process-data#shared/process-data#g'
```

`# ponytail: adaface/facenet train scripts are reference-only; left as-is. If you
ever wire them into a test, fix their imports then, not now.`

- [ ] **Step 6: Update `pyproject.toml` package config**

```toml
[tool.pytest.ini_options]
testpaths = ["models/arcface/tests", "application/tests", "evaluation/tests"]
asyncio_mode = "auto"

[tool.setuptools.packages.find]
where = ["."]
include = ["models*", "application*"]
```

- [ ] **Step 7: Verify the tree collects**

Run: `python -m pytest --collect-only -q`
Expected: collection succeeds (tests may be skipped for missing weights, but **no import errors**). If `ModuleNotFoundError: training_pipeline`, a `sed` target was missed — re-run Step 5's grep to find it.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: reorganize into models/ + shared/ for consolidation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Model registry

The single source of truth for every per-model difference. Pure data + three builder functions that take a loaded checkpoint dict and return a weights-loaded backbone. Fully testable with synthetic state dicts (no real weights).

**Files:**
- Create: `application/backend/registry.py`
- Test: `application/tests/test_registry.py`

**Interfaces:**
- Consumes: `models.arcface.model.FaceEmbedding`, `models.adaface.iresnet.iresnet50`, `facenet_pytorch.InceptionResnetV1`
- Produces:
  - `@dataclass ModelSpec(name, ckpt_filename, input_size, mean, std, ckpt_key, needs_cfg, threshold, use_tta, builder)`
  - `REGISTRY: dict[str, ModelSpec]` with keys `"arcface"`, `"adaface"`, `"facenet"`
  - `build_arcface(ckpt: dict) -> nn.Module`, `build_adaface(ckpt) -> nn.Module`, `build_facenet(ckpt) -> nn.Module` — each returns a backbone whose `forward(x)` is the raw 512-d embedding, with `state_dict` loaded `strict=True`

- [ ] **Step 1: Write the failing test**

```python
# application/tests/test_registry.py
import pytest
import torch
from application.backend.registry import REGISTRY, ModelSpec


def test_registry_has_three_models():
    assert set(REGISTRY) == {"arcface", "adaface", "facenet"}


def test_per_model_contracts_match_spec():
    arc, ada, fac = REGISTRY["arcface"], REGISTRY["adaface"], REGISTRY["facenet"]
    assert arc.input_size == 160 and arc.mean == (0.485, 0.456, 0.406)
    assert arc.threshold == 0.565 and arc.use_tta is False and arc.needs_cfg is True
    assert ada.input_size == 112 and ada.mean == (0.5, 0.5, 0.5) and ada.ckpt_key == "model"
    assert fac.input_size == 160 and fac.mean == (0.5, 0.5, 0.5)
    assert fac.ckpt_key == "model_state_dict"


def test_build_facenet_strict_load_rejects_wrong_keys():
    # Tensor value under a wrong key → must fail on strict key validation,
    # not earlier on a non-tensor value.
    with pytest.raises(RuntimeError):
        REGISTRY["facenet"].builder({"model_state_dict": {"not.a.real.key": torch.zeros(1)}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest application/tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: application.backend.registry`

- [ ] **Step 3: Write the registry**

```python
# application/backend/registry.py
"""Single source of truth for per-model inference contracts + backbone builders.

Each builder takes a loaded checkpoint dict and returns a backbone whose
forward(x) is the raw (un-normalized) 512-d embedding, weights loaded strict=True.
Normalization to unit length is added later by EmbedWrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt_filename: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    ckpt_key: str          # key under which the state_dict lives in the checkpoint
    needs_cfg: bool         # arcface stores embedding_dim under ckpt["cfg"]
    threshold: float
    builder: Callable[[dict], nn.Module]
    use_tta: bool = False


def build_arcface(ckpt: dict) -> nn.Module:
    from models.arcface.model import FaceEmbedding
    dim = ckpt["cfg"]["train"]["embedding_dim"]
    model = FaceEmbedding(embedding_dim=dim, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.eval()


def build_adaface(ckpt: dict) -> nn.Module:
    from models.adaface.iresnet import iresnet50
    sd = ckpt["model"]
    dim = sd["fc.weight"].shape[0]
    model = iresnet50(embedding_size=dim)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def build_facenet(ckpt: dict) -> nn.Module:
    # Mirror the notebook: backbone held as .bb so bb.-prefixed keys load directly.
    from facenet_pytorch import InceptionResnetV1

    class FaceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bb = InceptionResnetV1(pretrained=None, classify=False)

        def forward(self, x):
            return self.bb(x)  # raw embedding; EmbedWrapper normalizes

    model = FaceNet()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval()


REGISTRY: dict[str, ModelSpec] = {
    "arcface": ModelSpec("arcface", "arcface.pt", 160, *IMAGENET,
                         ckpt_key="model", needs_cfg=True, threshold=0.565,
                         builder=build_arcface, use_tta=False),
    "adaface": ModelSpec("adaface", "adaface.pt", 112, *HALF,
                         ckpt_key="model", needs_cfg=False, threshold=0.5,
                         builder=build_adaface),
    "facenet": ModelSpec("facenet", "facenet.pth", 160, *HALF,
                         ckpt_key="model_state_dict", needs_cfg=False, threshold=0.5,
                         builder=build_facenet),
}
```

Note: `ModelSpec(..., *IMAGENET, ...)` unpacks `(mean, std)` into the `mean`/`std` fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest application/tests/test_registry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add application/backend/registry.py application/tests/test_registry.py
git commit -m "feat(app): model registry with per-model contracts and builders

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: EmbedWrapper

One adapter that turns any raw-embedding backbone into a uniform `embed_normalized` / `embed_tta` interface. Removes per-model normalization duplication.

**Files:**
- Create: `application/backend/embed_wrapper.py`
- Test: `application/tests/test_embed_wrapper.py`

**Interfaces:**
- Produces: `EmbedWrapper(backbone: nn.Module)` with `.embed_normalized(x) -> Tensor` (unit-norm) and `.embed_tta(x) -> Tensor` (flip-averaged, unit-norm)

- [ ] **Step 1: Write the failing test**

```python
# application/tests/test_embed_wrapper.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest application/tests/test_embed_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: application.backend.embed_wrapper`

- [ ] **Step 3: Write the wrapper**

```python
# application/backend/embed_wrapper.py
"""Uniform embedding interface over any backbone returning raw 512-d vectors."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def embed_normalized(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), p=2, dim=1)

    def embed_tta(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.embed_normalized(x)
        e2 = self.embed_normalized(torch.flip(x, dims=[-1]))
        return F.normalize(e1 + e2, p=2, dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest application/tests/test_embed_wrapper.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add application/backend/embed_wrapper.py application/tests/test_embed_wrapper.py
git commit -m "feat(app): EmbedWrapper for uniform normalized/TTA embedding

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Registry-driven Embedder

Replace the ArcFace-only `Embedder` with one that loads any registry model from a checkpoint and embeds via `EmbedWrapper`.

**Files:**
- Modify: `application/backend/inference.py` (full rewrite)
- Test: `application/tests/test_inference.py` (keep the existing skip-if-missing integration test; add a registry-dispatch unit test)

**Interfaces:**
- Consumes: `REGISTRY`, `ModelSpec`, `EmbedWrapper`
- Produces: `Embedder(model_name: str, checkpoint: Path, device="cpu")` with attributes `.dim:int`, `.spec:ModelSpec`, and method `.embed(face_tensor) -> Tensor` (uses `embed_tta` iff `spec.use_tta`)

- [ ] **Step 1: Write the failing test (registry dispatch, no real weights)**

```python
# application/tests/test_inference.py  (append below existing integration test)
import torch
from application.backend import inference as inf


def test_embedder_uses_tta_flag(monkeypatch):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest application/tests/test_inference.py::test_embedder_uses_tta_flag -v`
Expected: FAIL (current `Embedder` has no `use_tta` set via this path / signature mismatch)

- [ ] **Step 3: Rewrite `inference.py`**

```python
# application/backend/inference.py
"""Load a registry model from a checkpoint and embed an aligned face tensor."""
from __future__ import annotations

from pathlib import Path

import torch

from .embed_wrapper import EmbedWrapper
from .registry import REGISTRY


class Embedder:
    def __init__(self, model_name: str, checkpoint: Path, device: str = "cpu") -> None:
        if model_name not in REGISTRY:
            raise ValueError(f"unknown model {model_name!r}; have {list(REGISTRY)}")
        self.spec = REGISTRY[model_name]
        self.device = device
        self.use_tta = self.spec.use_tta
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        backbone = self.spec.builder(ckpt).to(device)
        self.model = EmbedWrapper(backbone).to(device).eval()
        # Probe embedding dim with a dummy forward at the model's input size.
        s = self.spec.input_size
        with torch.no_grad():
            self.dim = int(self.model.embed_normalized(torch.zeros(1, 3, s, s, device=device)).shape[1])

    @torch.no_grad()
    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        fn = self.model.embed_tta if self.use_tta else self.model.embed_normalized
        return fn(face_tensor).squeeze(0).cpu()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest application/tests/test_inference.py::test_embedder_uses_tta_flag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add application/backend/inference.py application/tests/test_inference.py
git commit -m "feat(app): registry-driven Embedder with per-model TTA dispatch

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Per-model aligner

Parameterize the MTCNN aligner by input size + normalization so each model crops/normalizes correctly (ArcFace 160/ImageNet, AdaFace 112/0.5, FaceNet 160/0.5).

**Files:**
- Modify: `application/backend/face_align.py` (full rewrite)
- Test: `application/tests/test_face_align.py`

**Interfaces:**
- Produces: `FaceAligner(input_size: int, mean, std, device="cpu")` with `.align(img_bytes) -> Tensor | None` returning `(3, input_size, input_size)`

- [ ] **Step 1: Write the failing test**

```python
# application/tests/test_face_align.py
from application.backend.face_align import FaceAligner


def test_aligner_records_size_and_norm():
    a = FaceAligner(input_size=112, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert a.input_size == 112
    # MTCNN must be configured at the requested size.
    assert a.mtcnn.image_size == 112
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest application/tests/test_face_align.py -v`
Expected: FAIL (current `FaceAligner.__init__` takes only `device`)

- [ ] **Step 3: Rewrite `face_align.py`**

```python
# application/backend/face_align.py
"""MTCNN-based face alignment, parameterized per model (size + normalization)."""
from __future__ import annotations

from io import BytesIO

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms


class FaceAligner:
    def __init__(self, input_size: int, mean, std, device: str = "cpu") -> None:
        self.device = device
        self.input_size = input_size
        self.mtcnn = MTCNN(
            image_size=input_size, margin=0, post_process=False,
            device=device, keep_all=False,
        )
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(list(mean), list(std)),
        ])

    def align(self, img_bytes: bytes) -> torch.Tensor | None:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        face = self.mtcnn(img)
        if face is None:
            return None
        # facenet-pytorch returns float in [0,255]; convert via PIL for normalize.
        face_np = face.byte().permute(1, 2, 0).cpu().numpy()
        return self.normalize(Image.fromarray(face_np))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest application/tests/test_face_align.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add application/backend/face_align.py application/tests/test_face_align.py
git commit -m "feat(app): per-model aligner (input size + normalization)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Multi-model app backend (`main.py`)

Load every available checkpoint at startup, expose `GET /models`, and route enroll/verify by a `model` field to per-model embedder + store, each with the version guard. `db.py` is unchanged (it already supports `embedding_version`).

**Files:**
- Modify: `application/backend/main.py` (full rewrite)
- Test: `application/tests/test_api.py` (replace fixture; add `/models` test that needs no weights)

**Interfaces:**
- Consumes: `REGISTRY`, `Embedder`, `FaceAligner`, `EnrollmentDB`
- Produces: HTTP `GET /models` → `[{name, available, input_size, reason}]`; `POST /enroll {name, image, model}`; `POST /verify {image, model, threshold?}`; per-model store at `embeddings/<model>/`

- [ ] **Step 1: Write the failing test (`/models` works with zero weights)**

```python
# application/tests/test_api.py  (replace the fixture + add this test)
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    from application.backend import main as app_main
    app_main._reset_for_tests()
    return TestClient(app_main.app)


def test_models_endpoint_lists_all_registry_entries(client):
    r = client.get("/models")
    assert r.status_code == 200
    names = {m["name"] for m in r.json()}
    assert names == {"arcface", "adaface", "facenet"}
    # No checkpoints present in tmp env → all unavailable, none crash.
    assert all(m["available"] is False for m in r.json())


def test_verify_unavailable_model_returns_503(client):
    r = client.post("/verify", json={"image": "", "model": "arcface"})
    assert r.status_code == 503
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest application/tests/test_api.py::test_models_endpoint_lists_all_registry_entries -v`
Expected: FAIL (no `/models` route; old `main.py` is single-model)

- [ ] **Step 3: Rewrite `main.py`**

```python
# application/backend/main.py
"""FastAPI app: multi-model enroll / verify / list / delete + static frontend."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import EnrollmentDB
from .face_align import FaceAligner
from .inference import Embedder
from .registry import REGISTRY

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "application" / "frontend"

app = FastAPI(title="Face Recognition")

# name -> {"embedder", "aligner", "db", "threshold"} for available models only
_loaded: dict[str, dict] = {}
_default_model: str | None = None


def _config() -> dict:
    """Read env on every reset so tests can monkeypatch between resets."""
    return {
        "models_dir": Path(os.environ.get("APP_MODELS_DIR", ROOT / "application" / "models")),
        "data_dir": Path(os.environ.get("APP_DATA_DIR", ROOT / "application" / "embeddings")),
        "default_model": os.environ.get("APP_DEFAULT_MODEL"),  # None → first loaded
    }


def _embedding_version(checkpoint: Path, name: str, use_tta: bool) -> str:
    h = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{name}-{h.hexdigest()[:16]}-tta{int(use_tta)}"


def _threshold_for(name: str, default: float) -> float:
    return float(os.environ.get(f"APP_THRESHOLD_{name.upper()}", default))


def _reset_for_tests() -> None:
    """(Re)load every model whose checkpoint exists. Missing → not loaded."""
    global _loaded, _default_model
    cfg = _config()
    _loaded = {}
    for name, spec in REGISTRY.items():
        ckpt = cfg["models_dir"] / spec.ckpt_filename
        if not ckpt.exists():
            continue
        embedder = Embedder(name, ckpt, device="cpu")
        aligner = FaceAligner(spec.input_size, spec.mean, spec.std, device="cpu")
        version = _embedding_version(ckpt, name, spec.use_tta)
        db = EnrollmentDB(cfg["data_dir"] / name, dim=embedder.dim, embedding_version=version)
        _loaded[name] = {
            "embedder": embedder, "aligner": aligner, "db": db,
            "threshold": _threshold_for(name, spec.threshold),
        }
    # Resolve default: explicit env if loaded, else first loaded model, else None.
    want = cfg["default_model"]
    _default_model = want if want in _loaded else (next(iter(_loaded), None))


@app.on_event("startup")
def _startup() -> None:
    _reset_for_tests()


class EnrollReq(BaseModel):
    name: str
    image: str
    model: str | None = None


class VerifyReq(BaseModel):
    image: str
    model: str | None = None
    threshold: float | None = None


def _resolve(model: str | None) -> tuple[str, dict]:
    """Resolve an optional model name to (name, loaded_entry), applying the default."""
    name = model if model is not None else _default_model
    if name is None:
        raise HTTPException(503, "no models available — drop a checkpoint in application/models/")
    if name not in REGISTRY:
        raise HTTPException(400, f"unknown model {name!r}")
    if name not in _loaded:
        raise HTTPException(503, f"model {name!r} unavailable — checkpoint missing")
    return name, _loaded[name]


def _decode_align_embed(b64: str, m: dict):
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "invalid base64 image")
    tensor = m["aligner"].align(img_bytes)
    if tensor is None:
        raise HTTPException(422, "no face detected")
    return m["embedder"].embed(tensor)


@app.get("/models")
def list_models() -> list[dict]:
    out = []
    for name, spec in REGISTRY.items():
        avail = name in _loaded
        out.append({
            "name": name, "available": avail, "input_size": spec.input_size,
            "threshold": _loaded[name]["threshold"] if avail else spec.threshold,
            "is_default": name == _default_model,
            "reason": None if avail else "checkpoint missing",
        })
    return out


@app.post("/enroll")
def enroll(req: EnrollReq) -> dict:
    name, m = _resolve(req.model)
    emb = _decode_align_embed(req.image, m)
    eid = m["db"].enroll(req.name, emb)
    return {"id": eid, "name": req.name, "model": name}


@app.post("/verify")
def verify(req: VerifyReq) -> dict:
    name, m = _resolve(req.model)
    emb = _decode_align_embed(req.image, m)
    thr = req.threshold if req.threshold is not None else m["threshold"]
    result = m["db"].verify(emb, threshold=thr)
    result["model"] = name
    return result


@app.get("/enrolled")
def list_enrolled(model: str | None = None) -> list[dict]:
    name = model if model is not None else _default_model
    if name not in _loaded:
        return []
    return _loaded[name]["db"].list_enrolled()


@app.delete("/enrolled/{eid}")
def delete_enrolled(eid: str, model: str | None = None) -> dict:
    name = model if model is not None else _default_model
    if name not in _loaded:
        return {"ok": False}
    _loaded[name]["db"].delete(eid)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND), check_dir=False), name="static")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest application/tests/test_api.py -v`
Expected: the two new tests PASS; the pre-existing end-to-end test stays skipped (no checkpoint). If the old e2e test references the removed single-model `enroll` shape, update it to send `"model": "arcface"` and skip unless `application/models/arcface.pt` exists.

- [ ] **Step 5: Commit**

```bash
git add application/backend/main.py application/tests/test_api.py
git commit -m "feat(app): multi-model backend with /models, per-model stores + thresholds

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Frontend model selector

Add a dropdown populated from `GET /models`; send the selected model on enroll/verify/list; grey out unavailable models; seed the threshold control from the selected model.

**Files:**
- Modify: `application/frontend/index.html` (add `<select id="model-select">` in the header)
- Modify: `application/frontend/app.js` (fetch `/models`, include `model` in requests)

**Interfaces:**
- Consumes: `GET /models`, the `model` field on `/enroll` and `/verify`, `?model=` on `/enrolled`

- [ ] **Step 1: Add the selector to `index.html`**

Insert inside `<header class="bar">`, after the `.brand` div:

```html
      <label class="model-pick">
        <span class="field-label">Model</span>
        <select id="model-select"></select>
      </label>
```

Bump the cache-buster on the script tag: `<script src="/static/app.js?v=10"></script>`.

- [ ] **Step 2: Wire the selector in `app.js`**

Add near the top (after the existing `const` declarations):

```javascript
const modelSelect = document.getElementById('model-select');
const currentModel = () => modelSelect.value;

async function loadModels() {
  const res = await fetch('/models');
  const models = await res.json();
  modelSelect.innerHTML = '';
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m.name;
    opt.textContent = m.available ? m.name : `${m.name} (unavailable)`;
    opt.disabled = !m.available;
    modelSelect.appendChild(opt);
  }
  const firstAvail = models.find((m) => m.available);
  if (firstAvail) modelSelect.value = firstAvail.name;
}
loadModels();
```

In the enroll fetch body add `model: currentModel()`; in the verify fetch body add `model: currentModel()`; change the enrolled-list fetch to `fetch('/enrolled?model=' + encodeURIComponent(currentModel()))` and the delete fetch to append `?model=` similarly. Re-list on model change: `modelSelect.addEventListener('change', refreshList)` (use the existing list-refresh function name).

- [ ] **Step 3: Manual smoke (documented, no automated FE test)**

Run (only where weights + torch exist):
```bash
APP_MODELS_DIR=application/models uvicorn application.backend.main:app --port 8000
```
Open `http://localhost:8000`, confirm the dropdown lists three models with unavailable ones greyed out, and enroll/verify target the selected model. `# ponytail: frontend verified by manual smoke, no JS test harness in this repo.`

- [ ] **Step 4: Commit**

```bash
git add application/frontend/index.html application/frontend/app.js
git commit -m "feat(app): frontend model selector wired to /models

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Shared metric core (`metrics.py`)

The money path. Pure function from (sims, labels, threshold) to the canonical schema. Hand-checkable, no torch — this is the centerpiece that backfills Precision/Recall/F1 for ArcFace and FaceNet.

**Files:**
- Create: `evaluation/metrics.py`
- Create: `evaluation/__init__.py` (if missing), `evaluation/tests/__init__.py`
- Test: `evaluation/tests/test_metrics.py`

**Interfaces:**
- Produces: `compute_metrics(sims, labels, threshold) -> dict` with keys `{accuracy, precision, recall, f1, roc_auc, threshold, tp, fp, tn, fn}`; `cv_threshold_accuracy(sims, labels, n_folds=10) -> tuple[float, float, float]` returning `(mean_acc, std_acc, median_threshold)`

- [ ] **Step 1: Write the failing test (hand-computed values)**

```python
# evaluation/tests/test_metrics.py
import numpy as np
from evaluation.metrics import compute_metrics, cv_threshold_accuracy


def test_perfect_separation():
    m = compute_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 0, 2, 0)
    assert m["accuracy"] == 1.0 and m["precision"] == 1.0
    assert m["recall"] == 1.0 and m["f1"] == 1.0


def test_mixed_case_hand_computed():
    # preds at thr .5: [1,0,1,0]; labels [1,1,0,0]
    # idx0 tp, idx1 fn, idx2 fp, idx3 tn  -> acc .5 p .5 r .5 f1 .5
    m = compute_metrics([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0], threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5
    assert m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["threshold"] == 0.5


def test_cv_threshold_recovers_separating_threshold():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.6, 0.9, 200)
    neg = rng.uniform(0.1, 0.4, 200)
    sims = np.concatenate([pos, neg])
    labels = np.array([1] * 200 + [0] * 200)
    mean_acc, std_acc, thr = cv_threshold_accuracy(sims, labels)
    assert mean_acc > 0.95 and 0.4 <= thr <= 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest evaluation/tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: evaluation.metrics`

- [ ] **Step 3: Write `metrics.py`**

```python
# evaluation/metrics.py
"""Canonical verification metrics from (cosine sims, labels, threshold)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(sims, labels, threshold: float) -> dict:
    sims = np.asarray(sims, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (sims > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n = len(labels)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    try:
        roc_auc = float(roc_auc_score(labels, sims))
    except ValueError:           # only one class present
        roc_auc = 0.0
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def cv_threshold_accuracy(sims, labels, n_folds: int = 10):
    """10-fold CV: tune threshold on train folds, score held-out. Returns
    (mean_acc, std_acc, median_threshold)."""
    sims = np.asarray(sims, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(sims)
    folds = n_folds if n >= n_folds else 1
    fold_size = n // folds
    cand = np.linspace(-1, 1, 401)
    accs, thresholds = [], []
    for f in range(folds):
        lo, hi = f * fold_size, (f + 1) * fold_size if f < folds - 1 else n
        val = np.zeros(n, dtype=bool)
        val[lo:hi] = True
        train = ~val if folds > 1 else val
        best_a, best_t = 0.0, 0.0
        for t in cand:
            a = ((sims[train] > t).astype(int) == labels[train]).mean()
            if a > best_a:
                best_a, best_t = a, float(t)
        accs.append(float(((sims[val] > best_t).astype(int) == labels[val]).mean()))
        thresholds.append(best_t)
    return float(np.mean(accs)), float(np.std(accs)), float(np.median(thresholds))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest evaluation/tests/test_metrics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py evaluation/__init__.py evaluation/tests/
git commit -m "feat(eval): canonical compute_metrics + CV threshold tuning

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Per-model evaluator CLI (`evaluate.py`)

Embed an LFW pair list using each model's **registry contract** (size + norm + alignment), tune the threshold via CV, write `results/<model>.json` in the canonical schema. This is the script the user runs where weights + LFW data + torch exist.

**Files:**
- Create: `evaluation/evaluate.py`
- Create: `evaluation/pairs.py` (LFW pair loading — lift `load_pairs_txt` from `models/arcface/eval_lfw.py`)
- Test: `evaluation/tests/test_evaluate.py` (tests the embedding-to-metrics path with a stub embedder; no real weights)

**Interfaces:**
- Consumes: `REGISTRY`, `compute_metrics`, `cv_threshold_accuracy`, `EmbedWrapper`/`Embedder`
- Produces: `evaluate_from_sims(name, sims, labels) -> dict` (canonical schema; confusion metrics AND accuracy both at the CV threshold — see Step 3); CLI `python -m evaluation.evaluate --model <name> --pairs <txt> --aligned-root <dir> [--raw-root <dir>] [--out <json>]`
- **Eval preprocessing protocol:** aligned images are loaded **directly** (resize to the model's `input_size` + per-model normalize) — MTCNN is **not** re-run on already-aligned crops (that would shift/drop faces and break comparability with the branch metrics). A missing aligned file falls back to an 80%-center-crop of the raw image, matching the existing ArcFace evaluator. MTCNN alignment belongs only to the live app path (raw camera frames), not here.

- [ ] **Step 1: Write the failing test (sims→metrics path, stubbed embeddings)**

```python
# evaluation/tests/test_evaluate.py
import numpy as np
from evaluation.evaluate import evaluate_from_sims


def test_evaluate_from_sims_emits_canonical_schema():
    sims = np.concatenate([np.linspace(0.6, 0.9, 50), np.linspace(0.1, 0.4, 50)])
    labels = np.array([1] * 50 + [0] * 50)
    out = evaluate_from_sims("arcface", sims, labels)
    assert set(out) >= {"accuracy", "precision", "recall", "f1", "roc_auc",
                        "threshold", "std_acc", "tp", "fp", "tn", "fn", "model"}
    assert out["model"] == "arcface"
    assert 0.0 <= out["f1"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest evaluation/tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: evaluation.evaluate`

- [ ] **Step 3: Write `pairs.py` and `evaluate.py`**

```python
# evaluation/pairs.py
"""LFW pairs.txt loader (lifted from the ArcFace eval path)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LfwPair:
    name1: str
    idx1: int
    name2: str
    idx2: int
    same: int


def load_pairs_txt(pairs_txt: Path) -> list[LfwPair]:
    lines = Path(pairs_txt).read_text().strip().splitlines()
    pairs: list[LfwPair] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) == 3:        # same person: name idx1 idx2
            pairs.append(LfwPair(parts[0], int(parts[1]), parts[0], int(parts[2]), 1))
        elif len(parts) == 4:      # different: name1 idx1 name2 idx2
            pairs.append(LfwPair(parts[0], int(parts[1]), parts[2], int(parts[3]), 0))
    return pairs
```

```python
# evaluation/evaluate.py
"""Per-model LFW evaluation -> canonical metrics JSON.

Embeds pairs by loading ALREADY-ALIGNED images directly (resize to the model's
input size + per-model normalization). MTCNN is NOT re-run on aligned crops —
that belongs to the live app path only, and re-detecting inside a crop would
shift/drop faces and break comparability with the branch metrics. Missing aligned
files fall back to an 80%-center-crop of the raw image (matches the existing
ArcFace evaluator). Threshold tuned by 10-fold CV; all metrics reported at it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from application.backend.registry import REGISTRY
from .metrics import compute_metrics, cv_threshold_accuracy


def evaluate_from_sims(model: str, sims, labels) -> dict:
    # CV chooses the operating threshold; report ALL metrics (incl. accuracy) at it
    # so accuracy and tp/fp/tn/fn come from the same predictions (internally consistent).
    _, std_acc, thr = cv_threshold_accuracy(sims, labels)
    out = compute_metrics(sims, labels, thr)   # accuracy here == (tp+tn)/n at thr
    out["std_acc"] = std_acc                    # spread across folds, for reference
    out["model"] = model
    out["n_pairs"] = int(len(labels))
    return out


def _make_transform(spec):
    """Resize to the model's input size + per-model normalization. No detection."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((spec.input_size, spec.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(list(spec.mean), list(spec.std)),
    ])


def _load_aligned_or_crop(aligned: Path, raw: Path | None, tf):
    """Load aligned image directly; if missing, 80%-center-crop the raw image."""
    from PIL import Image
    if aligned.exists():
        return tf(Image.open(aligned).convert("RGB"))
    if raw is None or not raw.exists():
        raise FileNotFoundError(f"missing aligned {aligned} and no raw fallback")
    img = Image.open(raw).convert("RGB")
    w, h = img.size
    side = int(0.8 * min(w, h))
    l, t = (w - side) // 2, (h - side) // 2
    return tf(img.crop((l, t, l + side, t + side)))


def _embed_pairs(model: str, pairs, aligned_root: Path, raw_root: Path | None, device: str):
    """Embed every referenced image at the model's contract; return (sims, labels)."""
    import torch
    from application.backend.inference import Embedder

    spec = REGISTRY[model]
    ckpt = Path("application/models") / spec.ckpt_filename
    embedder = Embedder(model, ckpt, device=device)
    tf = _make_transform(spec)
    cache: dict[tuple[str, int], "torch.Tensor"] = {}

    def _emb(name: str, idx: int):
        key = (name, idx)
        if key in cache:
            return cache[key]
        fname = f"{name}_{idx:04d}.jpg"
        aligned = aligned_root / name / fname
        raw = (raw_root / name / fname) if raw_root else None
        tensor = _load_aligned_or_crop(aligned, raw, tf)
        e = embedder.embed(tensor)          # embed() normalizes + applies TTA per spec
        cache[key] = e
        return e

    sims, labels = [], []
    for p in pairs:
        e1, e2 = _emb(p.name1, p.idx1), _emb(p.name2, p.idx2)
        sims.append(float((e1 * e2).sum()))
        labels.append(p.same)
    return np.array(sims), np.array(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY))
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--aligned-root", required=True)
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from .pairs import load_pairs_txt
    pairs = load_pairs_txt(Path(args.pairs))
    sims, labels = _embed_pairs(
        args.model, pairs, Path(args.aligned_root),
        Path(args.raw_root) if args.raw_root else None, args.device,
    )
    result = evaluate_from_sims(args.model, sims, labels)
    out = Path(args.out) if args.out else Path("evaluation/results") / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

`# ponytail: aligned images loaded direct + resize/normalize; MTCNN stays in the
app path only. 80%-center-crop fallback mirrors the existing ArcFace evaluator.`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest evaluation/tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/evaluate.py evaluation/pairs.py evaluation/tests/test_evaluate.py
git commit -m "feat(eval): per-model LFW evaluator with registry-driven preprocessing

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Comparison aggregator (`compare.py`)

Read all `results/<model>.json` and emit a single `comparison.md` + `comparison.csv` so the four required metrics sit side-by-side across models.

**Files:**
- Create: `evaluation/compare.py`
- Test: `evaluation/tests/test_compare.py`

**Interfaces:**
- Consumes: canonical-schema dicts from `results/*.json`
- Produces: `build_comparison(results: list[dict]) -> tuple[str, str]` returning `(markdown, csv)` with columns `model, accuracy, precision, recall, f1, roc_auc, threshold`

- [ ] **Step 1: Write the failing test**

```python
# evaluation/tests/test_compare.py
from evaluation.compare import build_comparison


def test_comparison_table_has_all_models_and_metrics():
    results = [
        {"model": "arcface", "accuracy": 0.909, "precision": 0.9, "recall": 0.92,
         "f1": 0.91, "roc_auc": 0.95, "threshold": 0.565},
        {"model": "adaface", "accuracy": 0.994, "precision": 0.996, "recall": 0.991,
         "f1": 0.994, "roc_auc": 0.999, "threshold": 0.237},
    ]
    md, csv = build_comparison(results)
    assert "arcface" in md and "adaface" in md
    assert "| model |" in md.lower() or "model" in md.splitlines()[0]
    assert csv.splitlines()[0] == "model,accuracy,precision,recall,f1,roc_auc,threshold"
    assert "arcface,0.909" in csv
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest evaluation/tests/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: evaluation.compare`

- [ ] **Step 3: Write `compare.py`**

```python
# evaluation/compare.py
"""Aggregate per-model result JSONs into one comparison table (md + csv)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]


def build_comparison(results: list[dict]) -> tuple[str, str]:
    rows = sorted(results, key=lambda r: r["model"])

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)

    header = "| " + " | ".join(COLS) + " |"
    sep = "|" + "|".join(["---"] * len(COLS)) + "|"
    md_lines = [header, sep]
    csv_lines = [",".join(COLS)]
    for r in rows:
        cells = [r["model"]] + [fmt(r.get(c, "")) for c in COLS[1:]]
        md_lines.append("| " + " | ".join(cells) + " |")
        csv_lines.append(",".join(cells))
    return "\n".join(md_lines) + "\n", "\n".join(csv_lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="evaluation/results")
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    results = [json.loads(p.read_text()) for p in rdir.glob("*.json")
               if p.name not in {"comparison.json"}]
    md, csv = build_comparison(results)
    (rdir / "comparison.md").write_text(md)
    (rdir / "comparison.csv").write_text(csv)
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest evaluation/tests/test_compare.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/compare.py evaluation/tests/test_compare.py
git commit -m "feat(eval): comparison table aggregator (md + csv)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Docs, gitignore, full-suite gate

Update `.gitignore` for the new weights drop-zone, write the README (how to place weights, run the app, run evals — plus the unauthenticated-app flag), and run the whole test suite.

**Files:**
- Modify: `.gitignore`, `README.md`
- Create: `application/models/.gitkeep`

- [ ] **Step 1: Update `.gitignore`**

Replace the stale training-artifact paths (they referenced `training-pipeline/`) and add per-model stores + weights drop-zone:

```gitignore
# Training artifacts
models/*/checkpoints/
models/*/tensorboard_logs/
models/*/*.log

# App runtime
application/embeddings/
application/models/*.pt
application/models/*.pth
```

```bash
touch application/models/.gitkeep
git add -f application/models/.gitkeep
```

- [ ] **Step 2: Write the README consolidation section**

Append to `README.md`:

```markdown
## Models & model selection

Three trained models live side-by-side under `models/`: ArcFace (ResNet50, 160px,
ImageNet norm), AdaFace (IResNet50, 112px, 0.5 norm), FaceNet (InceptionResnetV1,
160px, 0.5 norm). They produce non-comparable embeddings, so each has its own
enrollment store under `application/embeddings/<model>/`.

### Run the app
Drop checkpoints into `application/models/` named `arcface.pt`, `adaface.pt`,
`facenet.pth` (gitignored). The app loads whatever is present; missing models show
as unavailable in the dropdown.

```bash
uvicorn application.backend.main:app --port 8000
```

Per-model threshold override: `APP_THRESHOLD_ARCFACE=0.6 uvicorn ...`.

> Security: this app is unauthenticated — it is a local demo. Do not expose it to
> an untrusted network without adding access control.

### Evaluate (Accuracy / Precision / Recall / F1 / ROC-AUC)
Needs the checkpoint + LFW pairs/images + a torch environment.

```bash
python -m evaluation.evaluate --model arcface --pairs <pairs.txt> --aligned-root <dir>
python -m evaluation.evaluate --model adaface --pairs <pairs.txt> --aligned-root <dir>
python -m evaluation.evaluate --model facenet --pairs <pairs.txt> --aligned-root <dir>
python -m evaluation.compare          # writes evaluation/results/comparison.{md,csv}
```

AdaFace's extended 8-suite InsightFace report is preserved at
`evaluation/adaface_evaluation.ipynb` (AdaFace-only, not part of the 3-model table).
```

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: all non-skipped tests PASS; weight-dependent integration tests SKIP with clear reasons. No import errors, no collection errors.

- [ ] **Step 4: Commit**

```bash
git add .gitignore README.md application/models/.gitkeep
git commit -m "docs: weights drop-zone, model-selection + eval instructions

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 5: Push the branch (only when the user asks)**

```bash
git push -u origin feat/model-consolidation
```

---

## Notes for the implementer

- **Environment reality:** torch/torchvision/facenet-pytorch are required project
  deps and must be installed (Task 1 Step 1b) — the unit tests import them. What
  the tests do **not** need is a GPU or any model weights: every unit test
  (registry, EmbedWrapper, Embedder dispatch, aligner config, metrics, evaluator
  sims-path, comparison) runs on CPU with no checkpoints. The weight-dependent
  paths (app enroll/verify e2e, `evaluate.py` end-to-end over real LFW images) stay
  skipped until the user supplies checkpoints + data. The original exploration box
  had no torch and only FaceNet's weights (`/Users/n3m0/Downloads/facenet_best.pth`);
  run the plan somewhere torch installs.
- **No retraining anywhere.** P/R/F1 for ArcFace and FaceNet come purely from
  re-scoring existing embeddings through `compute_metrics`. AdaFace has no trainer
  in the repo — only its inference backbone (`iresnet.py`) and eval are carried over.
- **Merge to `main`:** open a PR from `feat/model-consolidation`; do not push to `main` directly.
