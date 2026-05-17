# Face Recognition with Siamese Network — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, evaluate, and deploy a Siamese face-recognition system on GCP, with a FastAPI web app for enroll/verify via browser webcam.

**Architecture:** Local repo holds code + infra; GCP A100 VM downloads datasets, trains a ResNet50 + triplet-loss embedding network on CASIA-WebFace + CelebA, evaluates on LFW, uploads checkpoint to GCS, and self-terminates. App runs locally with CPU inference against the trained checkpoint, storing enrollments in SQLite + .npy.

**Tech Stack:** Python 3.11, PyTorch 2.x + torchvision, facenet-pytorch (MTCNN), FastAPI, Uvicorn, SQLite, numpy, pandas, pyarrow, TensorBoard, Kaggle CLI, gcloud, Docker (optional).

**Spec:** `docs/superpowers/specs/2026-05-17-face-recognition-siamese-design.md`

---

## File Structure

**Local (committed):**

| Path | Purpose |
|------|---------|
| `pyproject.toml` | Project deps + tool config |
| `.gitignore` | Exclude data/checkpoints/secrets |
| `README.md` | Top-level overview |
| `preprocess-data/check.py` | Dataset sanity-check script |
| `preprocess-data/README.md` | Dataset stats |
| `process-data/process.py` | MTCNN alignment + manifest builder |
| `process-data/data_processing.ipynb` | Processing report |
| `training-pipeline/src/__init__.py` | Package marker |
| `training-pipeline/src/model.py` | `FaceEmbedding` (ResNet50 + 512-d head) |
| `training-pipeline/src/dataset.py` | `TripletDataset` + `PKSampler` |
| `training-pipeline/src/loss.py` | `batch_hard_triplet_loss` |
| `training-pipeline/src/eval_lfw.py` | In-loop LFW evaluator |
| `training-pipeline/src/utils.py` | Seeding, AverageMeter, etc. |
| `training-pipeline/src/train.py` | Main entry + auto-stop |
| `training-pipeline/configs/train.yaml` | Hyperparameters |
| `training-pipeline/training_results.ipynb` | Training report |
| `training-pipeline/tests/test_loss.py` | Triplet loss unit tests |
| `training-pipeline/tests/test_model.py` | Model shape tests |
| `training-pipeline/tests/test_dataset.py` | PKSampler test |
| `training-pipeline/tests/test_smoke.py` | 2-epoch synthetic dataset smoke |
| `evaluation/eval_lfw.py` | Final 10-fold LFW benchmark |
| `evaluation/evaluation_results.ipynb` | Eval report |
| `application/backend/__init__.py` | Package marker |
| `application/backend/face_align.py` | MTCNN wrapper |
| `application/backend/inference.py` | Model load + embed |
| `application/backend/db.py` | SQLite + .npy storage |
| `application/backend/main.py` | FastAPI app |
| `application/frontend/index.html` | UI |
| `application/frontend/app.js` | Webcam + fetch |
| `application/frontend/style.css` | Styling (light/dark) |
| `application/tests/test_db.py` | Storage roundtrip |
| `application/tests/test_inference.py` | Embedding shape + idempotence |
| `application/tests/test_api.py` | FastAPI integration |
| `application/requirements.txt` | App deps |
| `application/Dockerfile` | Container image |
| `infra/create_vm.sh` | gcloud VM create + GCS bucket |
| `infra/setup_vm.sh` | VM-side env + dataset download |
| `infra/run_training.sh` | tmux launcher |
| `infra/README.md` | Operator runbook |

**Gitignored (created at runtime):**
- `preprocess-data/{casia-webface,celeba,lfw}/`
- `process-data/{train,val,lfw_pairs}/`, `process-data/manifest.parquet`, `process-data/alignment_failures.csv`
- `training-pipeline/checkpoints/`, `training-pipeline/tensorboard_logs/`
- `application/embeddings/`, `application/models/`

---

## Phase A — Local Repo Bootstrap

### Task 1: Bootstrap `.gitignore` + `pyproject.toml` + README

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.venv/
venv/
.pytest_cache/
.ipynb_checkpoints/
*.egg-info/

# Data (large, downloaded at runtime)
preprocess-data/casia-webface/
preprocess-data/celeba/
preprocess-data/lfw/
process-data/train/
process-data/val/
process-data/lfw_pairs/
process-data/manifest.parquet
process-data/alignment_failures.csv

# Training artifacts
training-pipeline/checkpoints/
training-pipeline/tensorboard_logs/
training-pipeline/*.log

# App runtime
application/embeddings/
application/models/

# Secrets
**/.kaggle/
**/kaggle.json
.env

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "face-recognition-siamese"
version = "0.1.0"
description = "Face recognition with Siamese network — CASIA/CelebA training, LFW eval, FastAPI app."
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "torchvision>=0.17",
    "facenet-pytorch>=2.5",
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "pillow>=10.0",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "tensorboard>=2.16",
    "tqdm>=4.66",
    "pyyaml>=6.0",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "python-multipart>=0.0.9",
    "kaggle>=1.6",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "httpx>=0.27", "ruff>=0.3"]

[tool.pytest.ini_options]
testpaths = ["training-pipeline/tests", "application/tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
```

- [ ] **Step 3: Rewrite top-level `README.md`**

```markdown
# Face Recognition with Siamese Network

Siamese-network face recognition trained on CASIA-WebFace + CelebA, evaluated on LFW, served via a FastAPI web app that captures faces from the browser webcam.

See [design spec](docs/superpowers/specs/2026-05-17-face-recognition-siamese-design.md) for full context.

## Repo layout
- `preprocess-data/` — raw datasets + sanity-check script
- `process-data/` — MTCNN-aligned faces + analysis notebook
- `training-pipeline/` — model, loss, training entry, results notebook
- `evaluation/` — LFW 10-fold benchmark
- `application/` — FastAPI + browser-webcam UI
- `infra/` — GCP VM provisioning scripts

## Quickstart (after training)
```
pip install -e ".[dev]"
cd application && uvicorn backend.main:app --reload
# open http://localhost:8000
```

## Training (on GCP)
See `infra/README.md`.
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore pyproject.toml README.md
git commit -m "chore: bootstrap repo with deps, gitignore, README"
```

---

### Task 2: Infra scripts

**Files:**
- Create: `infra/create_vm.sh`
- Create: `infra/setup_vm.sh`
- Create: `infra/run_training.sh`
- Create: `infra/README.md`

- [ ] **Step 1: Write `infra/create_vm.sh`**

```bash
#!/usr/bin/env bash
# Create A100 VM + GCS bucket for training. Idempotent: skips existing resources.
set -euo pipefail

PROJECT="${PROJECT:-mk8s-sec-057aa9}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-siamese-train}"
BUCKET="${BUCKET:-${PROJECT}-siamese}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}"
ACCEL="${ACCEL:-type=nvidia-tesla-a100,count=1}"

gcloud config set project "$PROJECT"

# 1. GCS bucket for checkpoints.
if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  echo "Creating bucket gs://$BUCKET ..."
  gcloud storage buckets create "gs://$BUCKET" --location=us-central1
else
  echo "Bucket gs://$BUCKET exists."
fi

# 2. VM.
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" >/dev/null 2>&1; then
  echo "VM $VM_NAME exists. Starting if stopped..."
  gcloud compute instances start "$VM_NAME" --zone="$ZONE" || true
else
  echo "Creating VM $VM_NAME ..."
  gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="$ACCEL" \
    --image-family=common-cu124 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform
fi

echo "Done. SSH: gcloud compute ssh $VM_NAME --zone=$ZONE"
```

- [ ] **Step 2: Write `infra/setup_vm.sh`**

```bash
#!/usr/bin/env bash
# Runs ON the VM. Sets up Python env, Kaggle creds, downloads datasets.
set -euo pipefail

REPO_URL="${REPO_URL:-}"
KAGGLE_USERNAME="${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
KAGGLE_KEY="${KAGGLE_KEY:?set KAGGLE_KEY}"

cd "$HOME"

# 1. Clone repo if not present.
if [ ! -d IT4432E_Project ]; then
  test -n "$REPO_URL" || { echo "set REPO_URL"; exit 1; }
  git clone "$REPO_URL" IT4432E_Project
fi
cd IT4432E_Project

# 2. Python deps. Deep Learning VM has conda+PyTorch+CUDA preinstalled.
pip install -e ".[dev]"

# 3. Kaggle credentials.
mkdir -p "$HOME/.kaggle"
cat > "$HOME/.kaggle/kaggle.json" <<EOF
{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_KEY"}
EOF
chmod 600 "$HOME/.kaggle/kaggle.json"

# 4. Download datasets.
download() {
  local slug="$1" dest="$2"
  mkdir -p "$dest"
  if [ -z "$(ls -A "$dest")" ]; then
    kaggle datasets download -d "$slug" -p "$dest"
    (cd "$dest" && unzip -q -o "*.zip" && rm -f "*.zip")
  else
    echo "$dest already populated; skipping download."
  fi
}

download "debarghamitraroy/casia-webface" "preprocess-data/casia-webface"
download "jessicali9530/celeba-dataset"   "preprocess-data/celeba"
download "jessicali9530/lfw-dataset"      "preprocess-data/lfw"

echo "Setup complete."
```

- [ ] **Step 3: Write `infra/run_training.sh`**

```bash
#!/usr/bin/env bash
# Run on VM. Launches training in tmux, follows log.
set -euo pipefail

cd "$HOME/IT4432E_Project"

SESSION="train"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "python -m training-pipeline.src.train --config training-pipeline/configs/train.yaml 2>&1 | tee training-pipeline/train.log"

echo "Training started in tmux session '$SESSION'."
echo "Attach: tmux attach -t $SESSION"
echo "Tail:   tail -f training-pipeline/train.log"
```

- [ ] **Step 4: Write `infra/README.md`**

```markdown
# Infra Runbook

## Prereqs
- `gcloud` authenticated to project `mk8s-sec-057aa9`.
- A100 quota in `us-central1` (fallback: change `MACHINE_TYPE` / `ACCEL` env vars to L4: `g2-standard-8` / `type=nvidia-l4,count=1`).
- Kaggle username + API key (passed via env, never committed).

## 1. Provision VM + bucket (laptop)
```
bash infra/create_vm.sh
```

## 2. SSH to VM, set up env (one-time)
```
gcloud compute ssh siamese-train --zone=us-central1-a
# on VM:
git clone <REPO_URL> IT4432E_Project
cd IT4432E_Project
export KAGGLE_USERNAME=n3m09999
export KAGGLE_KEY=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export REPO_URL=<REPO_URL>
bash infra/setup_vm.sh
```

## 3. Process datasets (one-time)
```
python process-data/process.py
```

## 4. Train
```
bash infra/run_training.sh
tmux attach -t train
```

Training auto-stops the VM on completion.

## 5. Pull checkpoint (laptop)
```
gcloud storage cp gs://mk8s-sec-057aa9-siamese/checkpoints/best.pt application/models/
```

## Cost guardrails
- VM is stopped automatically on training success/failure.
- To manually stop: `gcloud compute instances stop siamese-train --zone=us-central1-a`.
- To delete entirely: `gcloud compute instances delete siamese-train --zone=us-central1-a`.
```

- [ ] **Step 5: `chmod` and commit**

```bash
chmod +x infra/create_vm.sh infra/setup_vm.sh infra/run_training.sh
git add infra/
git commit -m "feat(infra): GCP A100 provisioning + Kaggle dataset download scripts"
```

---

## Phase B — Provision GCP VM

### Task 3: Run VM provisioning + verify A100

**Files:** none (operational)

- [ ] **Step 1: Run `create_vm.sh` from laptop**

```bash
bash infra/create_vm.sh
```

Expected: bucket created or noted as existing; VM created. If A100 quota error, re-run with:
```bash
MACHINE_TYPE=g2-standard-8 ACCEL=type=nvidia-l4,count=1 bash infra/create_vm.sh
```

- [ ] **Step 2: SSH + confirm GPU visible**

```bash
gcloud compute ssh siamese-train --zone=us-central1-a --command='nvidia-smi'
```

Expected: NVIDIA-SMI output showing A100-SXM4-40GB (or L4 fallback).

- [ ] **Step 3: Push current repo to remote so VM can clone it**

If repo not yet on GitHub/GitLab:
```bash
# laptop
gh repo create IT4432E_Project --public --source=. --remote=origin --push
```
Or push to existing remote: `git push -u origin main`.

- [ ] **Step 4: No commit — operational task**

---

### Task 4: Setup VM + download datasets

**Files:** none (operational, executed on VM)

- [ ] **Step 1: SSH and bootstrap**

```bash
gcloud compute ssh siamese-train --zone=us-central1-a
```

On VM:
```bash
export KAGGLE_USERNAME=n3m09999
export KAGGLE_KEY=KGAT_1adbb9b10ef409c123325063aadf0914
export REPO_URL=<your remote URL>
git clone $REPO_URL IT4432E_Project
cd IT4432E_Project
bash infra/setup_vm.sh
```

- [ ] **Step 2: Verify dataset shapes**

```bash
du -sh preprocess-data/*
ls preprocess-data/casia-webface | head
ls preprocess-data/celeba | head
ls preprocess-data/lfw | head
```

Expected: CASIA ~10GB, CelebA ~1.5GB, LFW ~200MB. Each contains extracted folders.

- [ ] **Step 3: No commit — operational task**

---

## Phase C — Data Preprocessing & Processing

### Task 5: Dataset sanity-check script

**Files:**
- Create: `preprocess-data/check.py`
- Create: `preprocess-data/README.md` (placeholder, populated by script)
- Test: covered by running the script and inspecting output

- [ ] **Step 1: Write `preprocess-data/check.py`**

```python
"""Walks downloaded datasets, prints stats, writes preprocess-data/README.md."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
IMG_EXT = {".jpg", ".jpeg", ".png"}


def stat_casia() -> dict:
    base = ROOT / "casia-webface"
    # CASIA layout: casia-webface/<identity_id>/*.jpg (after unzip).
    # Find the directory of identities; some Kaggle dumps nest one level deeper.
    candidates = [base, *(p for p in base.iterdir() if p.is_dir())]
    for c in candidates:
        ids = [d for d in c.iterdir() if d.is_dir()]
        if len(ids) > 100:
            counts = Counter()
            for d in ids:
                counts[d.name] = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
            return {
                "root": str(c.relative_to(ROOT)),
                "identities": len(counts),
                "images": sum(counts.values()),
                "min_per_id": min(counts.values()),
                "max_per_id": max(counts.values()),
                "median_per_id": sorted(counts.values())[len(counts) // 2],
            }
    return {"error": "CASIA layout not recognized"}


def stat_celeba() -> dict:
    base = ROOT / "celeba"
    identity_file = next(base.rglob("identity_CelebA.txt"), None)
    img_dir = next(base.rglob("img_align_celeba"), None)
    if identity_file is None or img_dir is None:
        return {"error": "CelebA missing identity_CelebA.txt or img_align_celeba/"}
    counts = Counter()
    with identity_file.open() as f:
        for line in f:
            _, ident = line.strip().split()
            counts[ident] += 1
    n_imgs = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXT)
    return {
        "identity_file": str(identity_file.relative_to(ROOT)),
        "img_dir": str(img_dir.relative_to(ROOT)),
        "identities": len(counts),
        "images_labeled": sum(counts.values()),
        "images_on_disk": n_imgs,
        "min_per_id": min(counts.values()),
        "max_per_id": max(counts.values()),
    }


def stat_lfw() -> dict:
    base = ROOT / "lfw"
    pairs = next(base.rglob("pairs.txt"), None)
    deep = next(base.rglob("lfw-deepfunneled"), None) or next(base.rglob("lfw_funneled"), None)
    if pairs is None or deep is None:
        return {"error": "LFW missing pairs.txt or lfw-deepfunneled/"}
    pairs_lines = pairs.read_text().splitlines()
    return {
        "pairs_file": str(pairs.relative_to(ROOT)),
        "img_root": str(deep.relative_to(ROOT)),
        "pairs_count": len(pairs_lines) - 1,  # first line is fold/count header
        "identities_on_disk": sum(1 for d in deep.iterdir() if d.is_dir()),
    }


def main() -> None:
    report = {
        "casia": stat_casia(),
        "celeba": stat_celeba(),
        "lfw": stat_lfw(),
    }
    print(json.dumps(report, indent=2))
    readme = ROOT / "README.md"
    readme.write_text(
        "# Datasets\n\nAuto-generated by `check.py`.\n\n"
        "```json\n" + json.dumps(report, indent=2) + "\n```\n"
        "\nRaw data is **gitignored**; run `infra/setup_vm.sh` to recreate.\n"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on VM after datasets are downloaded**

```bash
cd ~/IT4432E_Project && python preprocess-data/check.py
```

Expected: prints JSON. CASIA shows ~10K identities, ~500K images. CelebA shows ~10K identities, ~200K images. LFW shows ~6000 pairs.

- [ ] **Step 3: Commit (laptop, after pulling generated README)**

On VM: `git add preprocess-data/check.py preprocess-data/README.md && git commit -m "feat(data): dataset sanity check + generated stats README" && git push`.

---

### Task 6: MTCNN alignment + manifest builder

**Files:**
- Create: `process-data/process.py`
- Test: covered by smoke run

- [ ] **Step 1: Write `process-data/process.py`**

```python
"""MTCNN-align all faces, build train/val manifest, prepare LFW pairs."""
from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "preprocess-data"
OUT = ROOT / "process-data"
IMG_EXT = {".jpg", ".jpeg", ".png"}

log = logging.getLogger("process")


def build_pairs_casia() -> list[tuple[Path, str]]:
    base = next((RAW / "casia-webface").rglob("*"), None)
    casia_root = next(
        p for p in (RAW / "casia-webface").iterdir() if p.is_dir() and len(list(p.iterdir())) > 100
    )
    pairs = []
    for ident_dir in casia_root.iterdir():
        if not ident_dir.is_dir():
            continue
        for img in ident_dir.iterdir():
            if img.suffix.lower() in IMG_EXT:
                pairs.append((img, f"casia_{ident_dir.name}"))
    return pairs


def build_pairs_celeba() -> list[tuple[Path, str]]:
    identity_file = next((RAW / "celeba").rglob("identity_CelebA.txt"))
    img_dir = next((RAW / "celeba").rglob("img_align_celeba"))
    img_to_id = {}
    with identity_file.open() as f:
        for line in f:
            img, ident = line.strip().split()
            img_to_id[img] = f"celeba_{ident}"
    pairs = []
    for img in img_dir.iterdir():
        if img.name in img_to_id:
            pairs.append((img, img_to_id[img.name]))
    return pairs


def build_pairs_lfw() -> list[tuple[Path, str]]:
    deep = next((p for p in [RAW / "lfw"] for c in p.rglob("lfw-deepfunneled")), None) \
        or next((p for p in [RAW / "lfw"] for c in p.rglob("lfw_funneled")))
    deep = next((RAW / "lfw").rglob("lfw-deepfunneled"), None) or next((RAW / "lfw").rglob("lfw_funneled"))
    pairs = []
    for ident_dir in deep.iterdir():
        if ident_dir.is_dir():
            for img in ident_dir.iterdir():
                if img.suffix.lower() in IMG_EXT:
                    pairs.append((img, f"lfw_{ident_dir.name}"))
    return pairs


def align_batch(mtcnn: MTCNN, src_paths: list[Path], out_paths: list[Path]) -> list[bool]:
    imgs = []
    valid_idx = []
    for i, p in enumerate(src_paths):
        try:
            imgs.append(Image.open(p).convert("RGB"))
            valid_idx.append(i)
        except Exception:
            pass
    if not imgs:
        return [False] * len(src_paths)
    aligned = mtcnn(imgs, save_path=[str(out_paths[i]) for i in valid_idx])
    ok = [False] * len(src_paths)
    for j, i in enumerate(valid_idx):
        ok[i] = aligned[j] is not None
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-per-id", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit-per-id", type=int, default=80,
                    help="Cap images per identity (control train size for time budget).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device %s", device)

    mtcnn = MTCNN(image_size=160, margin=0, post_process=False, device=device, keep_all=False)

    train_root = OUT / "train"
    val_root = OUT / "val"
    lfw_root = OUT / "lfw_pairs"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)
    lfw_root.mkdir(parents=True, exist_ok=True)

    failures_path = OUT / "alignment_failures.csv"
    failures = open(failures_path, "w", newline="")
    fw = csv.writer(failures); fw.writerow(["source_path", "reason"])

    manifest_rows = []

    def process_split(name: str, pairs: list[tuple[Path, str]], split_for_train: bool):
        # Cap per identity.
        by_id: dict[str, list[Path]] = {}
        for p, ident in pairs:
            by_id.setdefault(ident, []).append(p)
        for ident in list(by_id):
            random.shuffle(by_id[ident])
            by_id[ident] = by_id[ident][: args.limit_per_id]
            if len(by_id[ident]) < args.min_per_id:
                del by_id[ident]

        for ident, paths in tqdm(by_id.items(), desc=f"align {name}"):
            n = len(paths)
            n_val = max(1, int(n * args.val_frac)) if split_for_train else 0
            for idx in range(0, n, args.batch):
                batch_paths = paths[idx: idx + args.batch]
                out_paths = []
                for j, sp in enumerate(batch_paths):
                    global_idx = idx + j
                    if split_for_train and global_idx < n_val:
                        d = val_root / ident
                    elif split_for_train:
                        d = train_root / ident
                    else:
                        d = lfw_root / ident.removeprefix("lfw_")
                    d.mkdir(parents=True, exist_ok=True)
                    out_paths.append(d / sp.name)
                ok = align_batch(mtcnn, batch_paths, out_paths)
                for sp, op, ok_flag, j in zip(batch_paths, out_paths, ok, range(len(batch_paths))):
                    global_idx = idx + j
                    split = ("val" if split_for_train and global_idx < n_val else
                             "train" if split_for_train else "lfw")
                    if ok_flag:
                        manifest_rows.append({
                            "path": str(op.relative_to(ROOT)),
                            "identity_id": ident,
                            "source": name,
                            "split": split,
                        })
                    else:
                        fw.writerow([str(sp.relative_to(ROOT)), "mtcnn_no_face"])

    log.info("Loading CASIA pairs...")
    process_split("casia", build_pairs_casia(), split_for_train=True)
    log.info("Loading CelebA pairs...")
    process_split("celeba", build_pairs_celeba(), split_for_train=True)
    log.info("Loading LFW pairs...")
    process_split("lfw", build_pairs_lfw(), split_for_train=False)

    failures.close()

    df = pd.DataFrame(manifest_rows)
    df.to_parquet(OUT / "manifest.parquet", index=False)
    log.info("Wrote %d rows to manifest.parquet", len(df))
    log.info("Train: %d  Val: %d  LFW: %d",
             (df["split"] == "train").sum(),
             (df["split"] == "val").sum(),
             (df["split"] == "lfw").sum())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on VM**

```bash
cd ~/IT4432E_Project && python process-data/process.py 2>&1 | tee process-data/process.log
```

Expected: ~1–2 hours on A100. Final log shows train ~400K, val ~50K, lfw ~13K aligned images. Alignment failure rate < 10%.

- [ ] **Step 3: Commit script (laptop)**

```bash
git add process-data/process.py
git commit -m "feat(data): MTCNN alignment + manifest builder"
git push
```

---

### Task 7: Data processing notebook

**Files:**
- Create: `process-data/data_processing.ipynb`

- [ ] **Step 1: Write notebook (paste into a new .ipynb cell-by-cell, then save)**

Use this as `data_processing.ipynb` content (one Python cell per `---`):

```python
# Cell 1: Imports + load manifest
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

ROOT = Path("..").resolve()
df = pd.read_parquet(ROOT / "process-data/manifest.parquet")
print(df["split"].value_counts())
print("Unique train identities:", df[df.split=="train"]["identity_id"].nunique())
df.head()
```

```python
# Cell 2: Images per identity histogram
counts = df[df.split.isin(["train","val"])].groupby("identity_id").size()
plt.figure(figsize=(8,4))
plt.hist(counts.values, bins=50)
plt.xlabel("Images per identity"); plt.ylabel("# identities")
plt.title(f"Images per identity (median {int(counts.median())}, max {counts.max()})")
plt.show()
```

```python
# Cell 3: Sample grid per dataset
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
for row, src in enumerate(["casia", "celeba", "lfw"]):
    sample = df[df.source==src].sample(5, random_state=0)
    for col, p in enumerate(sample["path"]):
        img = Image.open(ROOT / p)
        axes[row, col].imshow(img); axes[row, col].axis("off")
    axes[row, 0].set_ylabel(src)
plt.tight_layout(); plt.show()
```

```python
# Cell 4: Alignment failure rate
fail = pd.read_csv(ROOT / "process-data/alignment_failures.csv") if (ROOT / "process-data/alignment_failures.csv").exists() else pd.DataFrame()
total_attempts = len(df) + len(fail)
print(f"Aligned: {len(df)}, Failed: {len(fail)}, Rate: {len(fail)/total_attempts:.2%}")
```

- [ ] **Step 2: Execute notebook on VM and download**

On VM:
```bash
jupyter nbconvert --to notebook --execute process-data/data_processing.ipynb --inplace
```

- [ ] **Step 3: Commit (laptop, after pulling)**

```bash
git pull
git add process-data/data_processing.ipynb
git commit -m "feat(data): data processing analysis notebook"
git push
```

---

## Phase D — Training Pipeline

### Task 8: Model (TDD)

**Files:**
- Create: `training-pipeline/src/__init__.py` (empty)
- Create: `training-pipeline/src/model.py`
- Test: `training-pipeline/tests/test_model.py`

- [ ] **Step 1: Write failing test**

`training-pipeline/tests/test_model.py`:
```python
import torch
from training_pipeline.src.model import FaceEmbedding


def test_embedding_shape_and_norm():
    m = FaceEmbedding(embedding_dim=512).eval()
    x = torch.randn(4, 3, 160, 160)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (4, 512)
    norms = y.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)
```

We need the package importable as `training_pipeline`. Add a small shim — see Step 2.

- [ ] **Step 2: Make package importable**

Create `training-pipeline/__init__.py` (empty). Add to `pyproject.toml` under `[tool.setuptools.packages.find]`:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["training-pipeline*", "application*"]

[tool.setuptools.package-dir]
training_pipeline = "training-pipeline"
application = "application"
```

Reinstall: `pip install -e .`.

- [ ] **Step 3: Run test — expect ImportError**

```bash
pytest training-pipeline/tests/test_model.py -v
```

Expected: FAIL (ModuleNotFoundError or AttributeError — model not implemented).

- [ ] **Step 4: Write `training-pipeline/src/model.py`**

```python
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FaceEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)
```

- [ ] **Step 5: Run test — expect PASS**

```bash
pytest training-pipeline/tests/test_model.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add training-pipeline/src/__init__.py training-pipeline/__init__.py training-pipeline/src/model.py training-pipeline/tests/test_model.py pyproject.toml
git commit -m "feat(model): FaceEmbedding ResNet50 backbone with L2-normalized 512-d head"
```

---

### Task 9: Triplet loss (TDD)

**Files:**
- Create: `training-pipeline/src/loss.py`
- Test: `training-pipeline/tests/test_loss.py`

- [ ] **Step 1: Write failing test**

`training-pipeline/tests/test_loss.py`:
```python
import torch
from training_pipeline.src.loss import batch_hard_triplet_loss


def test_loss_zero_when_anchor_equals_positive_and_far_from_neg():
    # 2 identities, 2 samples each. Same-id samples identical; different-id far apart.
    emb = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(emb, labels, margin=0.3)
    # Hardest pos dist = 0, hardest neg dist = sqrt(2) > margin → loss = 0.
    assert loss.item() == 0.0


def test_loss_positive_when_negatives_closer_than_positives():
    emb = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],  # same id 0, far from anchor
        [0.1, 0.1],  # different id, close to anchor
        [0.1, 0.1],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(emb, labels, margin=0.3)
    assert loss.item() > 0.0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest training-pipeline/tests/test_loss.py -v
```

- [ ] **Step 3: Write `training-pipeline/src/loss.py`**

```python
import torch
import torch.nn.functional as F


def _pairwise_dist(emb: torch.Tensor) -> torch.Tensor:
    # emb: (N, D), assumed L2-normalized but works either way.
    # squared euclidean for numerical stability, then sqrt.
    dot = emb @ emb.t()
    sq = (emb * emb).sum(dim=1)
    d2 = sq.unsqueeze(0) + sq.unsqueeze(1) - 2 * dot
    d2 = d2.clamp(min=0.0)
    return torch.sqrt(d2 + 1e-12)


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """BatchHard triplet loss (Hermans et al., 2017)."""
    dist = _pairwise_dist(emb)
    n = labels.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same & ~eye
    neg_mask = ~same

    # Hardest positive: max distance among same-label (excl. self).
    pos_dist = dist.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    # Hardest negative: min distance among different-label.
    neg_dist = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values

    valid = (pos_dist != float("-inf")) & (neg_dist != float("inf"))
    losses = F.relu(pos_dist[valid] - neg_dist[valid] + margin)
    return losses.mean() if losses.numel() > 0 else torch.tensor(0.0, device=emb.device)
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add training-pipeline/src/loss.py training-pipeline/tests/test_loss.py
git commit -m "feat(train): BatchHard triplet loss"
```

---

### Task 10: PKSampler + TripletDataset (TDD)

**Files:**
- Create: `training-pipeline/src/dataset.py`
- Test: `training-pipeline/tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

`training-pipeline/tests/test_dataset.py`:
```python
from collections import Counter

from training_pipeline.src.dataset import PKSampler


def test_pk_sampler_yields_pk_batch_with_p_unique_ids():
    # 10 identities, 5 samples each. Labels at indices = identity_id.
    labels = [i // 5 for i in range(50)]  # 10 IDs × 5 samples.
    sampler = PKSampler(labels, p=4, k=2, num_batches=3, seed=0)
    batches = list(iter(sampler))
    assert len(batches) == 3
    for batch in batches:
        assert len(batch) == 8  # P*K
        ids = [labels[i] for i in batch]
        c = Counter(ids)
        assert len(c) == 4
        for v in c.values():
            assert v == 2
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Write `training-pipeline/src/dataset.py`**

```python
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FaceDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, transform=None):
        self.df = pd.read_parquet(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        ids = sorted(self.df["identity_id"].unique())
        self.id_to_idx = {i: n for n, i in enumerate(ids)}
        self.df["label"] = self.df["identity_id"].map(self.id_to_idx)
        self.labels = self.df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(ROOT / row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


class PKSampler(Sampler[list[int]]):
    """Yield batches of P identities × K samples each."""

    def __init__(self, labels: Sequence[int], p: int, k: int, num_batches: int, seed: int = 0):
        self.labels = list(labels)
        self.p = p
        self.k = k
        self.num_batches = num_batches
        self.seed = seed
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.by_label[lbl].append(idx)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        label_pool = [lbl for lbl, idxs in self.by_label.items() if len(idxs) >= self.k]
        for _ in range(self.num_batches):
            chosen_labels = rng.sample(label_pool, self.p)
            batch: list[int] = []
            for lbl in chosen_labels:
                batch.extend(rng.sample(self.by_label[lbl], self.k))
            yield batch

    def __len__(self):
        return self.num_batches
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add training-pipeline/src/dataset.py training-pipeline/tests/test_dataset.py
git commit -m "feat(train): PKSampler + FaceDataset with augmentations"
```

---

### Task 11: LFW in-loop evaluator

**Files:**
- Create: `training-pipeline/src/eval_lfw.py`

- [ ] **Step 1: Write `training-pipeline/src/eval_lfw.py`**

```python
"""Evaluate cosine-similarity verification accuracy on LFW pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .dataset import eval_transform

ROOT = Path(__file__).resolve().parents[2]


def load_pairs_txt(pairs_txt: Path) -> list[tuple[Path, Path, int]]:
    """Returns list of (img1, img2, same) where same in {0,1}."""
    lines = pairs_txt.read_text().splitlines()
    header = lines[0].split()
    n_folds, n_per_fold = int(header[0]), int(header[1])
    img_root = ROOT / "process-data" / "lfw_pairs"
    pairs = []
    i = 1
    for _ in range(n_folds):
        for _ in range(n_per_fold):
            parts = lines[i].split(); i += 1
            name, a, b = parts[0], int(parts[1]), int(parts[2])
            p1 = img_root / name / f"{name}_{a:04d}.jpg"
            p2 = img_root / name / f"{name}_{b:04d}.jpg"
            pairs.append((p1, p2, 1))
        for _ in range(n_per_fold):
            parts = lines[i].split(); i += 1
            n1, a, n2, b = parts[0], int(parts[1]), parts[2], int(parts[3])
            p1 = img_root / n1 / f"{n1}_{a:04d}.jpg"
            p2 = img_root / n2 / f"{n2}_{b:04d}.jpg"
            pairs.append((p1, p2, 0))
    return pairs


class _PairImgDataset(Dataset):
    def __init__(self, unique_paths: list[Path]):
        self.paths = unique_paths
        self.tf = eval_transform()

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def evaluate_lfw(model, pairs: list[tuple[Path, Path, int]], device: str, batch_size: int = 128,
                 max_pairs: int | None = None) -> dict:
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    # Deduplicate paths.
    unique = list({p for a, b, _ in pairs for p in (a, b)})
    path_to_idx = {p: i for i, p in enumerate(unique)}
    ds = _PairImgDataset(unique)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    model.eval()
    embs = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        embs.append(model(x).cpu())
    embs = torch.cat(embs, dim=0)  # already L2-normalized

    sims = []
    labels = []
    for a, b, same in pairs:
        ea = embs[path_to_idx[a]]
        eb = embs[path_to_idx[b]]
        sims.append(float((ea * eb).sum()))
        labels.append(same)
    sims_np = np.asarray(sims); labels_np = np.asarray(labels)

    # 10-fold threshold CV.
    n = len(sims_np)
    fold_size = n // 10
    accs = []
    for f in range(10):
        val_lo, val_hi = f * fold_size, (f + 1) * fold_size
        val_mask = np.zeros(n, dtype=bool); val_mask[val_lo:val_hi] = True
        train_mask = ~val_mask
        # Pick best threshold on train.
        cand = np.linspace(-1, 1, 401)
        best_acc, best_t = 0.0, 0.0
        for t in cand:
            pred = (sims_np[train_mask] > t).astype(int)
            acc = (pred == labels_np[train_mask]).mean()
            if acc > best_acc:
                best_acc, best_t = acc, t
        pred = (sims_np[val_mask] > best_t).astype(int)
        accs.append((pred == labels_np[val_mask]).mean())

    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "threshold_global": float(np.median([
            np.linspace(-1, 1, 401)[
                np.argmax([((sims_np > t).astype(int) == labels_np).mean()
                           for t in np.linspace(-1, 1, 401)])
            ]
        ])),
        "n_pairs": n,
    }
```

- [ ] **Step 2: Commit (no unit test — exercised by smoke test in Task 13)**

```bash
git add training-pipeline/src/eval_lfw.py
git commit -m "feat(eval): LFW pairs 10-fold cosine-threshold evaluator"
```

---

### Task 12: Utils (seeding + meters)

**Files:**
- Create: `training-pipeline/src/utils.py`

- [ ] **Step 1: Write `training-pipeline/src/utils.py`**

```python
import os
import random
import subprocess

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self): self.n = 0; self.sum = 0.0
    def update(self, v: float, n: int = 1):
        self.sum += v * n; self.n += n
    @property
    def avg(self) -> float:
        return self.sum / self.n if self.n else 0.0


def upload_to_gcs(local_path: str, gcs_uri: str) -> None:
    subprocess.run(["gcloud", "storage", "cp", local_path, gcs_uri], check=True)


def stop_vm_self() -> None:
    """If running on a GCE VM, stop self. Safe no-op otherwise."""
    try:
        name = subprocess.check_output(["hostname"], text=True).strip()
        # Discover zone from metadata.
        zone = subprocess.check_output([
            "curl", "-s", "-H", "Metadata-Flavor: Google",
            "http://metadata.google.internal/computeMetadata/v1/instance/zone"
        ], text=True).strip().split("/")[-1]
        subprocess.run(["gcloud", "compute", "instances", "stop", name,
                        f"--zone={zone}", "--quiet"], check=False)
    except Exception as e:
        print(f"VM auto-stop skipped: {e}")
```

- [ ] **Step 2: Commit**

```bash
git add training-pipeline/src/utils.py
git commit -m "feat(train): utils — seeding, AverageMeter, GCS upload, VM auto-stop"
```

---

### Task 13: Training entry + smoke test (TDD)

**Files:**
- Create: `training-pipeline/configs/train.yaml`
- Create: `training-pipeline/src/train.py`
- Test: `training-pipeline/tests/test_smoke.py`

- [ ] **Step 1: Write `training-pipeline/configs/train.yaml`**

```yaml
manifest: process-data/manifest.parquet
checkpoints_dir: training-pipeline/checkpoints
tensorboard_dir: training-pipeline/tensorboard_logs
gcs_bucket: gs://mk8s-sec-057aa9-siamese
seed: 42

train:
  epochs: 20
  p: 32
  k: 4
  batches_per_epoch: 1500
  margin: 0.3
  lr_head: 3.0e-4
  lr_backbone: 3.0e-5
  weight_decay: 1.0e-4
  warmup_steps: 500
  num_workers: 8
  embedding_dim: 512

eval:
  pairs_txt: preprocess-data/lfw/lfw_funneled/pairs.txt  # adjust to actual path discovered by check.py
  max_pairs_inloop: 1000  # subset for speed mid-training; full 6000 for final
  batch_size: 128

vm:
  auto_stop: true
```

- [ ] **Step 2: Write failing smoke test**

`training-pipeline/tests/test_smoke.py`:
```python
"""Two-epoch training on synthetic data must reduce loss."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from training_pipeline.src.train import run_training


def _make_synthetic_manifest(tmp: Path) -> Path:
    # 8 identities × 8 images. Random colored 64×64 images, but per-identity hue offset
    # so embedding can theoretically separate them.
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
            rows.append({"path": str(p), "identity_id": f"syn_{ident}",
                         "source": "syn", "split": split, "label": ident})
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
        "gcs_bucket": "",
        "seed": 0,
        "train": {
            "epochs": 2, "p": 4, "k": 2, "batches_per_epoch": 5,
            "margin": 0.3, "lr_head": 1e-3, "lr_backbone": 1e-4,
            "weight_decay": 0.0, "warmup_steps": 1, "num_workers": 0,
            "embedding_dim": 64,
        },
        "eval": {"pairs_txt": "", "max_pairs_inloop": 0, "batch_size": 4},
        "vm": {"auto_stop": False},
    }
    history = run_training(cfg)
    assert history["train_loss"][0] >= history["train_loss"][-1]
    assert (ckpt_dir / "last.pt").exists()
```

- [ ] **Step 3: Run — expect FAIL (no train.py yet)**

- [ ] **Step 4: Write `training-pipeline/src/train.py`**

```python
"""Train Siamese face embedding network."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import FaceDataset, PKSampler, train_transform, eval_transform
from .eval_lfw import evaluate_lfw, load_pairs_txt
from .loss import batch_hard_triplet_loss
from .model import FaceEmbedding
from .utils import AverageMeter, set_seed, stop_vm_self, upload_to_gcs

ROOT = Path(__file__).resolve().parents[2]


def _build_optimizer(model, lr_head: float, lr_backbone: float, wd: float):
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.fc" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ], weight_decay=wd)


def _cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def run_training(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(cfg["checkpoints_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(cfg["tensorboard_dir"]);  tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    train_ds = FaceDataset(Path(cfg["manifest"]), split="train", transform=train_transform())
    sampler = PKSampler(train_ds.labels, p=cfg["train"]["p"], k=cfg["train"]["k"],
                        num_batches=cfg["train"]["batches_per_epoch"], seed=cfg["seed"])
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=cfg["train"]["num_workers"],
                              pin_memory=(device == "cuda"),
                              persistent_workers=(cfg["train"]["num_workers"] > 0))

    model = FaceEmbedding(embedding_dim=cfg["train"]["embedding_dim"]).to(device)
    optim = _build_optimizer(model, cfg["train"]["lr_head"], cfg["train"]["lr_backbone"],
                             cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    total_steps = cfg["train"]["epochs"] * cfg["train"]["batches_per_epoch"]
    warmup = cfg["train"]["warmup_steps"]
    base_head = cfg["train"]["lr_head"]; base_back = cfg["train"]["lr_backbone"]

    # Optional LFW pairs for in-loop eval.
    pairs = []
    if cfg["eval"].get("pairs_txt"):
        pairs_path = ROOT / cfg["eval"]["pairs_txt"]
        if pairs_path.exists() and cfg["eval"]["max_pairs_inloop"] > 0:
            pairs = load_pairs_txt(pairs_path)[:cfg["eval"]["max_pairs_inloop"]]

    history = {"train_loss": [], "lfw_acc": []}
    best_acc = -1.0
    step = 0

    try:
        for epoch in range(cfg["train"]["epochs"]):
            model.train()
            meter = AverageMeter()
            for imgs, labels in tqdm(train_loader, desc=f"epoch {epoch}"):
                imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                lr_h = _cosine_lr(step, total_steps, warmup, base_head)
                lr_b = _cosine_lr(step, total_steps, warmup, base_back)
                optim.param_groups[0]["lr"] = lr_b
                optim.param_groups[1]["lr"] = lr_h

                optim.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    emb = model(imgs)
                    loss = batch_hard_triplet_loss(emb, labels, margin=cfg["train"]["margin"])
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update()

                meter.update(loss.item(), imgs.size(0))
                writer.add_scalar("train/loss_step", loss.item(), step)
                writer.add_scalar("train/lr_head", lr_h, step)
                step += 1

            history["train_loss"].append(meter.avg)
            writer.add_scalar("train/loss_epoch", meter.avg, epoch)

            acc = -1.0
            if pairs:
                metrics = evaluate_lfw(model, pairs, device, cfg["eval"]["batch_size"])
                acc = metrics["mean_acc"]
                writer.add_scalar("eval/lfw_acc", acc, epoch)
            history["lfw_acc"].append(acc)

            # Save.
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                       ckpt_dir / "last.pt")
            if acc > best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                           ckpt_dir / "best.pt")

            print(f"epoch {epoch}: loss={meter.avg:.4f} lfw_acc={acc:.4f}")

        # Upload to GCS.
        if cfg.get("gcs_bucket"):
            for name in ("best.pt", "last.pt"):
                p = ckpt_dir / name
                if p.exists():
                    upload_to_gcs(str(p), f"{cfg['gcs_bucket']}/checkpoints/{name}")
        # Dump history.
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
    finally:
        writer.close()
        if cfg["vm"].get("auto_stop") and device == "cuda":
            stop_vm_self()
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_training(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run smoke test — expect PASS**

```bash
pytest training-pipeline/tests/test_smoke.py -v -s
```

Expected: PASS in <30s on CPU.

- [ ] **Step 6: Commit**

```bash
git add training-pipeline/configs/train.yaml training-pipeline/src/train.py training-pipeline/tests/test_smoke.py
git commit -m "feat(train): training entry with mixed-precision, cosine LR, LFW in-loop eval, auto-stop"
```

---

### Task 14: Adjust eval config path to discovered LFW layout, kick off training

**Files:** `training-pipeline/configs/train.yaml`

- [ ] **Step 1: On VM, find LFW pairs path**

```bash
find preprocess-data/lfw -name 'pairs.txt'
```

- [ ] **Step 2: Update `eval.pairs_txt` in `configs/train.yaml`** to the discovered path. Commit + push.

- [ ] **Step 3: Pull on VM and launch training**

```bash
cd ~/IT4432E_Project && git pull && bash infra/run_training.sh
tmux attach -t train
# Detach: Ctrl-B then d.
```

Expected: loss starts ~0.5, drops toward 0.05 by epoch 20. LFW val acc reaches 0.95+.

- [ ] **Step 4: After training completes (or VM auto-stops), pull checkpoint to laptop**

```bash
mkdir -p application/models
gcloud storage cp gs://mk8s-sec-057aa9-siamese/checkpoints/best.pt application/models/best.pt
```

- [ ] **Step 5: No commit — operational; results notebook is next task**

---

### Task 15: Training results notebook

**Files:**
- Create: `training-pipeline/training_results.ipynb`

- [ ] **Step 1: Write notebook (Python cells, save as `.ipynb`)**

```python
# Cell 1: Load history + tensorboard summary
import json
from pathlib import Path
import matplotlib.pyplot as plt

hist = json.loads(Path("training-pipeline/checkpoints/history.json").read_text())
print(f"Epochs run: {len(hist['train_loss'])}")
print(f"Final train loss: {hist['train_loss'][-1]:.4f}")
print(f"Best LFW acc:   {max(hist['lfw_acc']):.4f}")
```

```python
# Cell 2: Curves
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
a.plot(hist["train_loss"]); a.set_title("Train loss"); a.set_xlabel("epoch")
b.plot(hist["lfw_acc"]);    b.set_title("LFW in-loop acc"); b.set_xlabel("epoch")
plt.tight_layout(); plt.show()
```

```python
# Cell 3: t-SNE of 1000 val embeddings
import torch
import numpy as np
from sklearn.manifold import TSNE
from training_pipeline.src.dataset import FaceDataset, eval_transform
from training_pipeline.src.model import FaceEmbedding
from torch.utils.data import DataLoader

ckpt = torch.load("training-pipeline/checkpoints/best.pt", map_location="cpu")
model = FaceEmbedding(embedding_dim=ckpt["cfg"]["train"]["embedding_dim"])
model.load_state_dict(ckpt["model"]); model.eval()

ds = FaceDataset(Path("process-data/manifest.parquet"), split="val", transform=eval_transform())
idxs = np.random.RandomState(0).choice(len(ds), size=1000, replace=False)
imgs = torch.stack([ds[i][0] for i in idxs])
labels = np.array([ds[i][1] for i in idxs])

with torch.no_grad():
    emb = model(imgs).cpu().numpy()

z = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(emb)
plt.figure(figsize=(8, 8))
plt.scatter(z[:, 0], z[:, 1], c=labels % 20, s=8, cmap="tab20")
plt.title("t-SNE of val embeddings (1000 samples, colored by identity mod 20)")
plt.show()
```

- [ ] **Step 2: Run notebook, commit output**

```bash
jupyter nbconvert --to notebook --execute training-pipeline/training_results.ipynb --inplace
git add training-pipeline/training_results.ipynb
git commit -m "feat(train): training results notebook with loss curves + t-SNE"
git push
```

---

## Phase E — Evaluation

### Task 16: Final LFW benchmark script

**Files:**
- Create: `evaluation/eval_lfw.py`

- [ ] **Step 1: Write `evaluation/eval_lfw.py`**

```python
"""Final 10-fold LFW evaluation on the best checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training_pipeline.src.eval_lfw import load_pairs_txt, evaluate_lfw
from training_pipeline.src.model import FaceEmbedding

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="application/models/best.pt")
    ap.add_argument("--pairs", default="preprocess-data/lfw/lfw_funneled/pairs.txt")
    ap.add_argument("--out", default="evaluation/results.json")
    args = ap.parse_args()

    ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu")
    model = FaceEmbedding(embedding_dim=ckpt["cfg"]["train"]["embedding_dim"])
    model.load_state_dict(ckpt["model"]); model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pairs = load_pairs_txt(ROOT / args.pairs)
    metrics = evaluate_lfw(model, pairs, device)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run (locally with downloaded checkpoint OR on VM before shutdown)**

```bash
python evaluation/eval_lfw.py
cat evaluation/results.json
```

Expected: `mean_acc >= 0.95`. If below, investigate (in spec — flag this).

- [ ] **Step 3: Commit**

```bash
git add evaluation/eval_lfw.py evaluation/results.json
git commit -m "feat(eval): final 10-fold LFW benchmark"
```

---

### Task 17: Evaluation results notebook

**Files:**
- Create: `evaluation/evaluation_results.ipynb`

- [ ] **Step 1: Write notebook**

```python
# Cell 1: Load metrics
import json
from pathlib import Path
m = json.loads(Path("evaluation/results.json").read_text())
print(json.dumps(m, indent=2))
```

```python
# Cell 2: Recompute pos/neg similarity distributions + ROC
import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from training_pipeline.src.eval_lfw import load_pairs_txt
from training_pipeline.src.model import FaceEmbedding
from training_pipeline.src.dataset import eval_transform
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ckpt = torch.load("application/models/best.pt", map_location="cpu")
model = FaceEmbedding(embedding_dim=ckpt["cfg"]["train"]["embedding_dim"])
model.load_state_dict(ckpt["model"]); model.eval()

pairs = load_pairs_txt(Path("preprocess-data/lfw/lfw_funneled/pairs.txt"))
unique = list({p for a,b,_ in pairs for p in (a,b)})
path_to_idx = {p:i for i,p in enumerate(unique)}

class D(Dataset):
    def __init__(self,paths): self.paths=paths; self.tf=eval_transform()
    def __len__(self): return len(self.paths)
    def __getitem__(self,i): return self.tf(Image.open(self.paths[i]).convert("RGB"))

dl = DataLoader(D(unique), batch_size=128, num_workers=4)
with torch.no_grad():
    embs = torch.cat([model(b) for b in dl])

sims = np.array([float((embs[path_to_idx[a]]*embs[path_to_idx[b]]).sum()) for a,b,_ in pairs])
labels = np.array([s for _,_,s in pairs])

plt.figure(figsize=(8,4))
plt.hist(sims[labels==1], bins=50, alpha=0.6, label="same")
plt.hist(sims[labels==0], bins=50, alpha=0.6, label="diff")
plt.legend(); plt.title("Cosine similarity distribution"); plt.xlabel("cos sim")
plt.show()

fpr,tpr,thr = roc_curve(labels, sims)
plt.figure(figsize=(5,5))
plt.plot(fpr,tpr,label=f"AUC={auc(fpr,tpr):.4f}")
plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("ROC"); plt.show()
```

```python
# Cell 3: Hardest false pos / false neg with image grids
order = np.argsort(sims)
# False neg: same-pair (label=1) with low sim.
fn_pairs = [pairs[i] for i in order if labels[i] == 1][:5]
# False pos: diff-pair (label=0) with high sim.
fp_pairs = [pairs[i] for i in order[::-1] if labels[i] == 0][:5]

def grid(pair_list, title):
    fig, axes = plt.subplots(len(pair_list), 2, figsize=(4, len(pair_list)*2))
    for i, (a, b, _) in enumerate(pair_list):
        axes[i, 0].imshow(Image.open(a)); axes[i, 0].axis("off")
        axes[i, 1].imshow(Image.open(b)); axes[i, 1].axis("off")
    fig.suptitle(title); plt.tight_layout(); plt.show()

grid(fn_pairs, "Hardest false negatives (same person, low similarity)")
grid(fp_pairs, "Hardest false positives (different people, high similarity)")
```

- [ ] **Step 2: Run + commit**

```bash
jupyter nbconvert --to notebook --execute evaluation/evaluation_results.ipynb --inplace
git add evaluation/evaluation_results.ipynb
git commit -m "feat(eval): evaluation results notebook with ROC, distributions, hardest cases"
```

---

## Phase F — Application

### Task 18: Face alignment + inference

**Files:**
- Create: `application/backend/__init__.py` (empty)
- Create: `application/backend/face_align.py`
- Create: `application/backend/inference.py`
- Test: `application/tests/test_inference.py`

- [ ] **Step 1: Write `application/backend/face_align.py`**

```python
from __future__ import annotations

from io import BytesIO

import torch
from facenet_pytorch import MTCNN
from PIL import Image

from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class FaceAligner:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.mtcnn = MTCNN(image_size=160, margin=0, post_process=False, device=device, keep_all=False)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def align(self, img_bytes: bytes) -> torch.Tensor | None:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        face = self.mtcnn(img)  # tensor (3,160,160) in [0,255] float, or None
        if face is None:
            return None
        # facenet-pytorch returns float in [0,255]; convert to PIL then normalize.
        face = face.byte().permute(1, 2, 0).cpu().numpy()
        return self.normalize(Image.fromarray(face))
```

- [ ] **Step 2: Write `application/backend/inference.py`**

```python
from __future__ import annotations

from pathlib import Path

import torch

from training_pipeline.src.model import FaceEmbedding


class Embedder:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(checkpoint, map_location=device)
        self.dim = ckpt["cfg"]["train"]["embedding_dim"]
        self.model = FaceEmbedding(embedding_dim=self.dim, pretrained=False).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        return self.model(face_tensor).squeeze(0).cpu()
```

- [ ] **Step 3: Write `application/tests/test_inference.py`**

```python
import io
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from application.backend.face_align import FaceAligner
from application.backend.inference import Embedder

# A real LFW image is needed; the test is skipped if the checkpoint or sample is missing.
import pytest

CKPT = Path("application/models/best.pt")
SAMPLE = next(Path("process-data/lfw_pairs").rglob("*.jpg"), None) if Path("process-data/lfw_pairs").exists() else None


@pytest.mark.skipif(not CKPT.exists() or SAMPLE is None, reason="checkpoint or sample missing")
def test_embedding_shape_and_idempotent():
    aligner = FaceAligner()
    embedder = Embedder(CKPT)
    img_bytes = SAMPLE.read_bytes()
    t = aligner.align(img_bytes)
    assert t is not None
    e1 = embedder.embed(t)
    e2 = embedder.embed(t)
    assert e1.shape == (embedder.dim,)
    assert torch.allclose(e1, e2)
    assert abs(e1.norm().item() - 1.0) < 1e-4
```

- [ ] **Step 4: Run tests — expect skip if no checkpoint, PASS once checkpoint exists**

```bash
pytest application/tests/test_inference.py -v
```

- [ ] **Step 5: Commit**

```bash
git add application/backend/__init__.py application/backend/face_align.py application/backend/inference.py application/tests/test_inference.py
git commit -m "feat(app): face alignment + embedding inference"
```

---

### Task 19: SQLite + .npy storage (TDD)

**Files:**
- Create: `application/backend/db.py`
- Test: `application/tests/test_db.py`

- [ ] **Step 1: Write failing test**

`application/tests/test_db.py`:
```python
from pathlib import Path
import numpy as np
import torch

from application.backend.db import EnrollmentDB


def test_enroll_verify_delete_roundtrip(tmp_path: Path):
    db = EnrollmentDB(tmp_path, dim=8)
    # Enroll two identities.
    id1 = db.enroll("alice", torch.tensor(np.array([1., 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)))
    id2 = db.enroll("bob",   torch.tensor(np.array([0., 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)))
    assert id1 != id2

    # Verify alice with a vector close to her.
    query = torch.tensor(np.array([0.99, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    query = query / query.norm()
    result = db.verify(query, threshold=0.9)
    assert result["matched"] is True
    assert result["best_match"] == "alice"
    assert result["score"] > 0.9

    # List.
    enrolled = db.list_enrolled()
    assert {e["name"] for e in enrolled} == {"alice", "bob"}

    # Delete bob.
    db.delete(id2)
    assert {e["name"] for e in db.list_enrolled()} == {"alice"}

    # Persist + reload.
    db2 = EnrollmentDB(tmp_path, dim=8)
    assert {e["name"] for e in db2.list_enrolled()} == {"alice"}


def test_verify_empty_returns_no_match(tmp_path: Path):
    db = EnrollmentDB(tmp_path, dim=8)
    q = torch.zeros(8); q[0] = 1.0
    result = db.verify(q, threshold=0.5)
    assert result["matched"] is False
    assert result["best_match"] is None
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Write `application/backend/db.py`**

```python
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import torch


class EnrollmentDB:
    def __init__(self, root: Path, dim: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        self.vec_path = self.root / "vectors.npy"
        self.dim = dim
        self._init_db()
        self._load_vectors()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS enrolled (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                row_idx INTEGER NOT NULL,
                created_at REAL NOT NULL
            )""")

    def _load_vectors(self):
        if self.vec_path.exists():
            arr = np.load(self.vec_path)
            if arr.ndim == 1: arr = arr.reshape(0, self.dim)
        else:
            arr = np.zeros((0, self.dim), dtype=np.float32)
        self._vectors = arr

    def _save_vectors(self):
        np.save(self.vec_path, self._vectors)

    def _compact(self):
        """Rebuild row_idx to remove deleted rows."""
        with sqlite3.connect(self.db_path) as c:
            rows = list(c.execute("SELECT id, row_idx FROM enrolled ORDER BY row_idx"))
            new_vecs = np.stack([self._vectors[r[1]] for r in rows]) if rows else np.zeros((0, self.dim), dtype=np.float32)
            self._vectors = new_vecs
            for new_idx, (eid, _) in enumerate(rows):
                c.execute("UPDATE enrolled SET row_idx=? WHERE id=?", (new_idx, eid))
        self._save_vectors()

    def enroll(self, name: str, emb: torch.Tensor) -> str:
        emb_np = emb.detach().cpu().numpy().astype(np.float32)
        assert emb_np.shape == (self.dim,)
        row_idx = self._vectors.shape[0]
        self._vectors = np.vstack([self._vectors, emb_np[None, :]])
        self._save_vectors()
        eid = uuid.uuid4().hex
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO enrolled(id,name,row_idx,created_at) VALUES (?,?,?,?)",
                      (eid, name, row_idx, time.time()))
        return eid

    def delete(self, eid: str) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute("DELETE FROM enrolled WHERE id=?", (eid,))
        self._compact()

    def list_enrolled(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as c:
            return [{"id": r[0], "name": r[1], "created_at": r[2]}
                    for r in c.execute("SELECT id, name, created_at FROM enrolled ORDER BY created_at")]

    def verify(self, emb: torch.Tensor, threshold: float) -> dict:
        if self._vectors.shape[0] == 0:
            return {"matched": False, "best_match": None, "score": 0.0, "threshold": threshold}
        q = emb.detach().cpu().numpy().astype(np.float32)
        sims = self._vectors @ q  # both L2-normalized -> cosine.
        # Aggregate per-name (max sim across enrollments of same name).
        with sqlite3.connect(self.db_path) as c:
            rows = list(c.execute("SELECT name, row_idx FROM enrolled"))
        best_score = -1.0; best_name = None
        for name, idx in rows:
            s = float(sims[idx])
            if s > best_score:
                best_score, best_name = s, name
        return {
            "matched": best_score >= threshold,
            "best_match": best_name,
            "score": best_score,
            "threshold": threshold,
        }
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add application/backend/db.py application/tests/test_db.py
git commit -m "feat(app): SQLite + .npy enrollment storage with cosine verify"
```

---

### Task 20: FastAPI app (TDD)

**Files:**
- Create: `application/backend/main.py`
- Create: `application/requirements.txt`
- Test: `application/tests/test_api.py`

- [ ] **Step 1: Write failing test**

`application/tests/test_api.py`:
```python
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from application.backend import main as app_main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_THRESHOLD", "-1.0")  # so any match passes
    # Reload module-level state.
    app_main._reset_for_tests()
    return TestClient(app_main.app)


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


@pytest.mark.skipif(
    not Path("application/models/best.pt").exists()
    or not list(Path("process-data/lfw_pairs").rglob("*.jpg"))[:2],
    reason="checkpoint or LFW sample missing"
)
def test_enroll_and_verify_end_to_end(client):
    samples = list(Path("process-data/lfw_pairs").rglob("*.jpg"))[:2]
    r = client.post("/enroll", json={"name": "test_user", "image": _img_b64(samples[0])})
    assert r.status_code == 200
    enrolled_id = r.json()["id"]

    r = client.post("/verify", json={"image": _img_b64(samples[0])})
    assert r.status_code == 200
    j = r.json()
    assert j["best_match"] == "test_user"

    r = client.get("/enrolled")
    assert any(e["name"] == "test_user" for e in r.json())

    r = client.delete(f"/enrolled/{enrolled_id}")
    assert r.status_code == 200
    r = client.get("/enrolled")
    assert all(e["name"] != "test_user" for e in r.json())
```

- [ ] **Step 2: Write `application/backend/main.py`**

```python
from __future__ import annotations

import base64
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import EnrollmentDB
from .face_align import FaceAligner
from .inference import Embedder

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "application" / "frontend"

app = FastAPI(title="Face Recognition")

_aligner: FaceAligner | None = None
_embedder: Embedder | None = None
_db: EnrollmentDB | None = None
_threshold: float = 0.5


def _config():
    return {
        "data_dir": Path(os.environ.get("APP_DATA_DIR", ROOT / "application" / "embeddings")),
        "checkpoint": Path(os.environ.get("APP_CKPT", ROOT / "application" / "models" / "best.pt")),
        "threshold": float(os.environ.get("APP_THRESHOLD", "0.5")),
    }


@app.on_event("startup")
def _startup():
    _reset_for_tests()  # idempotent init using current env.


def _reset_for_tests():
    """Re-init globals from env. Lets tests set APP_DATA_DIR/APP_THRESHOLD per test."""
    global _aligner, _embedder, _db, _threshold
    cfg = _config()
    _aligner = FaceAligner(device="cpu")
    if cfg["checkpoint"].exists():
        _embedder = Embedder(cfg["checkpoint"], device="cpu")
        _db = EnrollmentDB(cfg["data_dir"], dim=_embedder.dim)
    else:
        _embedder = None
        _db = None
    _threshold = cfg["threshold"]


class EnrollReq(BaseModel):
    name: str
    image: str  # base64


class VerifyReq(BaseModel):
    image: str


def _decode_align_embed(b64: str):
    if _embedder is None or _db is None:
        raise HTTPException(503, "Model not loaded. Place checkpoint at application/models/best.pt.")
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "invalid base64 image")
    tensor = _aligner.align(img_bytes)
    if tensor is None:
        raise HTTPException(422, "no face detected")
    return _embedder.embed(tensor)


@app.post("/enroll")
def enroll(req: EnrollReq):
    emb = _decode_align_embed(req.image)
    eid = _db.enroll(req.name, emb)
    return {"id": eid, "name": req.name}


@app.post("/verify")
def verify(req: VerifyReq):
    emb = _decode_align_embed(req.image)
    return _db.verify(emb, threshold=_threshold)


@app.get("/enrolled")
def list_enrolled():
    if _db is None: return []
    return _db.list_enrolled()


@app.delete("/enrolled/{eid}")
def delete_enrolled(eid: str):
    if _db is None: return {"ok": False}
    _db.delete(eid)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
```

- [ ] **Step 3: Write `application/requirements.txt`**

```
torch>=2.2
torchvision>=0.17
facenet-pytorch>=2.5
numpy>=1.26
pillow>=10.0
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.0
```

- [ ] **Step 4: Run test — expect PASS (or SKIP if no checkpoint yet)**

- [ ] **Step 5: Commit**

```bash
git add application/backend/main.py application/requirements.txt application/tests/test_api.py
git commit -m "feat(app): FastAPI endpoints — enroll / verify / list / delete"
```

---

### Task 21: Frontend (webcam capture)

**Files:**
- Create: `application/frontend/index.html`
- Create: `application/frontend/app.js`
- Create: `application/frontend/style.css`

- [ ] **Step 1: Write `application/frontend/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Face Recognition</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <header>
    <h1>Face Recognition</h1>
    <nav>
      <button data-tab="enroll" class="tab active">Enroll</button>
      <button data-tab="verify" class="tab">Verify</button>
      <button data-tab="list" class="tab">Enrolled</button>
    </nav>
  </header>

  <main>
    <section id="cam-section">
      <video id="cam" autoplay playsinline muted></video>
      <canvas id="snap" hidden></canvas>
    </section>

    <section id="enroll-section" class="panel">
      <input id="name" type="text" placeholder="Name" />
      <button id="enroll-btn">Capture & Enroll</button>
      <div id="enroll-result" class="result"></div>
    </section>

    <section id="verify-section" class="panel hidden">
      <button id="verify-btn">Capture & Verify</button>
      <div id="verify-result" class="result"></div>
    </section>

    <section id="list-section" class="panel hidden">
      <button id="refresh-btn">Refresh</button>
      <ul id="enrolled-list"></ul>
    </section>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `application/frontend/app.js`**

```javascript
const video = document.getElementById('cam');
const canvas = document.getElementById('snap');
const tabs = document.querySelectorAll('.tab');
const panels = {
  enroll: document.getElementById('enroll-section'),
  verify: document.getElementById('verify-section'),
  list:   document.getElementById('list-section'),
};

async function startCam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
}
startCam().catch(e => alert('Camera error: ' + e.message));

tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(x => x.classList.toggle('active', x === t));
  Object.entries(panels).forEach(([k, el]) => el.classList.toggle('hidden', k !== t.dataset.tab));
  if (t.dataset.tab === 'list') loadList();
}));

function snapBase64() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
}

document.getElementById('enroll-btn').addEventListener('click', async () => {
  const name = document.getElementById('name').value.trim();
  const el = document.getElementById('enroll-result');
  if (!name) { el.textContent = 'Enter a name'; return; }
  el.textContent = 'Capturing...';
  try {
    const r = await fetch('/enroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, image: snapBase64() }),
    });
    const j = await r.json();
    el.textContent = r.ok ? `Enrolled "${j.name}"` : `Error: ${j.detail}`;
  } catch (e) { el.textContent = 'Network error'; }
});

document.getElementById('verify-btn').addEventListener('click', async () => {
  const el = document.getElementById('verify-result');
  el.textContent = 'Capturing...';
  try {
    const r = await fetch('/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: snapBase64() }),
    });
    const j = await r.json();
    if (!r.ok) { el.textContent = `Error: ${j.detail}`; return; }
    if (j.matched) {
      el.innerHTML = `<strong class="match">Match: ${j.best_match}</strong><br />score: ${j.score.toFixed(3)} (threshold ${j.threshold})`;
    } else {
      el.innerHTML = `<strong class="nomatch">No match</strong><br />best: ${j.best_match ?? '—'} score ${j.score.toFixed(3)}`;
    }
  } catch (e) { el.textContent = 'Network error'; }
});

async function loadList() {
  const ul = document.getElementById('enrolled-list');
  ul.innerHTML = '<li>Loading...</li>';
  const r = await fetch('/enrolled');
  const items = await r.json();
  ul.innerHTML = '';
  for (const it of items) {
    const li = document.createElement('li');
    li.innerHTML = `<span>${it.name}</span> <button data-id="${it.id}">Delete</button>`;
    li.querySelector('button').addEventListener('click', async () => {
      await fetch(`/enrolled/${it.id}`, { method: 'DELETE' });
      loadList();
    });
    ul.appendChild(li);
  }
  if (!items.length) ul.innerHTML = '<li>No enrolled faces yet.</li>';
}

document.getElementById('refresh-btn').addEventListener('click', loadList);
```

- [ ] **Step 3: Write `application/frontend/style.css`**

```css
:root {
  --bg: #ffffff;
  --fg: #111111;
  --muted: #666;
  --accent: #0d6efd;
  --match: #198754;
  --nomatch: #dc3545;
  --panel-bg: #f6f7f9;
  --border: #d0d4dc;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1217;
    --fg: #f4f6f8;
    --muted: #9aa4b2;
    --accent: #4c8df6;
    --panel-bg: #161c24;
    --border: #2a313c;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.5;
}
header {
  padding: 1rem;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
  gap: 1rem;
}
h1 { margin: 0; font-size: 1.25rem; letter-spacing: -0.01em; }
nav { display: flex; gap: 0.5rem; }
.tab {
  border: 1px solid var(--border);
  background: var(--panel-bg);
  color: var(--fg);
  border-radius: 999px;
  padding: 0.5rem 1rem;
  cursor: pointer;
  min-height: 44px;
}
.tab.active { background: var(--accent); color: white; border-color: var(--accent); }
main { padding: 1rem; display: grid; gap: 1rem; max-width: 720px; margin: 0 auto; }
#cam {
  width: 100%; max-width: 480px; aspect-ratio: 4/3;
  background: black; border-radius: 12px; display: block; margin: 0 auto;
}
.panel { background: var(--panel-bg); padding: 1rem; border-radius: 12px; border: 1px solid var(--border); }
.panel.hidden { display: none; }
input[type="text"] {
  width: 100%;
  padding: 0.75rem;
  margin-bottom: 0.75rem;
  font-size: 1rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--fg);
  min-height: 44px;
}
button {
  padding: 0.75rem 1.25rem;
  font-size: 1rem;
  border: none;
  border-radius: 8px;
  background: var(--accent);
  color: white;
  cursor: pointer;
  min-height: 44px;
}
button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.result { margin-top: 0.75rem; font-size: 0.95rem; color: var(--muted); min-height: 2.5em; }
.match { color: var(--match); }
.nomatch { color: var(--nomatch); }
ul { list-style: none; padding: 0; margin: 1rem 0 0 0; }
li {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.5rem 0.75rem; border: 1px solid var(--border); border-radius: 8px;
  margin-bottom: 0.5rem; background: var(--bg);
}
```

- [ ] **Step 4: Manual test in browser**

```bash
cd application && pip install -r requirements.txt
APP_THRESHOLD=0.5 uvicorn backend.main:app --reload
# open http://localhost:8000 — grant camera, enroll yourself, verify
```

Expected: webcam preview shows; enrolling captures and stores; verify returns your name with score >0.5.

- [ ] **Step 5: Commit**

```bash
git add application/frontend/
git commit -m "feat(app): frontend — webcam capture + enroll/verify/list UI with light+dark theme"
```

---

### Task 22: Dockerfile (optional)

**Files:**
- Create: `application/Dockerfile`

- [ ] **Step 1: Write `application/Dockerfile`**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY application/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY training-pipeline /app/training-pipeline
COPY application /app/application

ENV APP_CKPT=/app/application/models/best.pt
ENV APP_DATA_DIR=/app/application/embeddings
ENV APP_THRESHOLD=0.5

EXPOSE 8000
CMD ["uvicorn", "application.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Commit**

```bash
git add application/Dockerfile
git commit -m "feat(app): Dockerfile for FastAPI deployment"
```

---

## Phase G — Final wrap-up

### Task 23: End-to-end smoke check + push

- [ ] **Step 1: Run full test suite locally**

```bash
pytest -v
```

Expected: all tests pass (or skip with clear reason if checkpoint not yet present locally).

- [ ] **Step 2: Confirm git tree is clean**

```bash
git status
```

- [ ] **Step 3: Push everything**

```bash
git push
```

- [ ] **Step 4: Verify VM is stopped (cost guardrail)**

```bash
gcloud compute instances list --filter='name=siamese-train'
```

Expected: STATUS=TERMINATED. If RUNNING and training is done, stop manually:
```bash
gcloud compute instances stop siamese-train --zone=us-central1-a
```

- [ ] **Step 5: Final commit (only if anything pending)**

---

## Self-Review

**Spec coverage check:**
- `/preprocess-data` raw + check + README → Tasks 4, 5. ✓
- `/process-data` aligned faces + manifest + notebook → Tasks 6, 7. ✓
- `/training-pipeline` model + loss + dataset + train + notebook → Tasks 8–15. ✓
- `/evaluation` LFW final benchmark + notebook → Tasks 16, 17. ✓
- `/application` FastAPI + frontend → Tasks 18–22. ✓
- GCP A100 spin-up + auto-stop → Tasks 2–4, train.py auto_stop. ✓
- Kaggle API on VM with token → Task 2 (`setup_vm.sh`), Task 4 (env vars). ✓
- Verify=1:N with score + threshold → `db.py` (Task 19), surfaced in `/verify` (Task 20). ✓
- SQLite + .npy storage → Task 19. ✓
- PyTorch + ResNet50 + triplet loss + online hard mining → Tasks 8, 9. ✓
- LFW eval ≥95% gate → noted in Task 16 Step 2. ✓
- One commit per logical stage → each task ends in commit. ✓

**Placeholder scan:** No TBD/TODO. Every code step has full code. Exact filepaths everywhere. Exact gcloud commands.

**Type consistency:** `FaceEmbedding(embedding_dim=...)` signature matches across model.py, train.py, inference.py. `EnrollmentDB(root, dim)` matches across db.py + main.py. `evaluate_lfw(model, pairs, device, ...)` matches across in-loop + final eval. ✓

**Ambiguity check:** LFW `pairs.txt` path is dataset-layout-dependent — Task 14 Step 1 explicitly handles via `find`. Threshold value in Task 20 default 0.5; final threshold from evaluation results documented in evaluation results notebook. ✓

Plan ready for execution.
