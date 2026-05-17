# Face Recognition with Siamese Network — Design Spec

**Date:** 2026-05-17
**Status:** Approved (pending spec review)
**Author:** Tạ Quốc Hùng (with Claude)

## Goal

Build a face-recognition system that:

1. Trains a Siamese embedding network on CASIA-WebFace + CelebA.
2. Evaluates the model on LFW using the standard 6000-pair protocol.
3. Ships a web application where a user can enroll a face (via browser webcam) and later verify whether a newly captured face matches any enrolled identity (1:N identification with score).

Training runs on a GCP A100 40GB VM with a target wall-clock of 4–5 hours and a budget of ~$20.

## Non-Goals

- Mobile native app.
- Liveness / anti-spoof detection (out of scope for v1).
- Production-grade auth, multi-tenant, or HTTPS termination beyond local demo.
- Distributed / multi-GPU training.
- ArcFace / CosFace heads — sticking with triplet loss for simplicity.

## Assumptions

- User has working `gcloud` CLI authenticated to project `mk8s-sec-057aa9`.
- A100 quota is available in `us-central1` (will check; fall back to L4 if not).
- Kaggle account credentials: username `n3m09999`, API token starting with `KGAT_...`. Token is never committed to git.
- Datasets are public and downloadable via Kaggle API:
  - `debarghamitraroy/casia-webface` (~10 GB, ~500K images, ~10K identities)
  - `jessicali9530/celeba-dataset` (~1.5 GB, ~200K images, ~10K identities)
  - `jessicali9530/lfw-dataset` (~200 MB, 13K images, 5749 identities, includes `pairs.txt`)
- Project repo is currently empty except `README.md`.

## Architecture

### Repo layout

```
IT4432E_Project/
├── preprocess-data/         # Raw datasets (gitignored except README)
│   ├── casia-webface/
│   ├── celeba/
│   ├── lfw/
│   └── README.md            # Dataset stats + usability notes
├── process-data/            # Aligned/cropped face tensors
│   ├── train/<identity_id>/*.jpg
│   ├── val/<identity_id>/*.jpg
│   ├── lfw_pairs/           # Pre-aligned pair tensors
│   ├── manifest.parquet     # All (path, identity, split) rows
│   └── data_processing.ipynb
├── training-pipeline/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── dataset.py       # PKSampler + TripletDataset
│   │   ├── model.py         # ResNet50 + embedding head
│   │   ├── loss.py          # BatchHard triplet loss
│   │   ├── train.py         # Main training entry
│   │   ├── eval_lfw.py      # In-loop LFW val evaluator
│   │   └── utils.py
│   ├── configs/train.yaml
│   ├── checkpoints/         # Gitignored; uploaded to GCS
│   ├── tensorboard_logs/    # Gitignored
│   └── training_results.ipynb
├── evaluation/
│   ├── eval_lfw.py          # Final LFW benchmark
│   └── evaluation_results.ipynb
├── application/
│   ├── backend/
│   │   ├── main.py          # FastAPI app
│   │   ├── inference.py     # Load model, embed
│   │   ├── db.py            # SQLite + .npy storage layer
│   │   └── face_align.py    # MTCNN wrapper
│   ├── frontend/
│   │   ├── index.html
│   │   ├── app.js           # getUserMedia + fetch
│   │   └── style.css
│   ├── models/              # Final inference checkpoint (gitignored)
│   ├── embeddings/          # SQLite + .npy enrolled data (gitignored)
│   ├── requirements.txt
│   └── Dockerfile
├── infra/
│   ├── create_vm.sh         # gcloud compute instances create ...
│   ├── setup_vm.sh          # Conda env + kaggle CLI + dataset DL
│   ├── run_training.sh      # tmux + train.py + GCS sync + auto-stop
│   └── README.md
├── docs/superpowers/specs/
│   └── 2026-05-17-face-recognition-siamese-design.md
├── .gitignore
├── pyproject.toml
└── README.md
```

### Stage 1 — Preprocess (`preprocess-data/`)

Run on GCP VM (not local — datasets too large).

Inputs: Kaggle credentials.
Outputs: raw extracted datasets on local SSD, `preprocess-data/README.md` with stats.

Steps:
1. `pip install kaggle`, place `kaggle.json` at `~/.kaggle/kaggle.json` with `chmod 600`.
2. `kaggle datasets download -d debarghamitraroy/casia-webface -p preprocess-data/casia-webface && unzip ...`
3. Same for `jessicali9530/celeba-dataset` and `jessicali9530/lfw-dataset`.
4. Sanity check script (`preprocess-data/check.py`):
   - Walk each dataset, count identities and images per identity.
   - Verify CASIA folder structure `casia-webface/<identity_id>/*.jpg`.
   - Verify CelebA has `identity_CelebA.txt` and `img_align_celeba/*.jpg`.
   - Verify LFW has `lfw-deepfunneled/<name>/*.jpg` and `pairs.txt`.
   - Print stats; flag any unusable subset.
5. Write `preprocess-data/README.md` with discovered stats and any issues.

### Stage 2 — Process (`process-data/`)

Inputs: `preprocess-data/`.
Outputs: aligned face tensors organized by identity, `manifest.parquet`, `data_processing.ipynb`.

Steps:
1. MTCNN face detect + align (`facenet-pytorch`'s `MTCNN(image_size=160, margin=0)`).
2. For each image: detect face, crop, resize to 160×160, save as JPEG (quality 95).
3. Identity dedup: prefix CelebA identity IDs with `celeba_` and CASIA with `casia_` to avoid collision.
4. Filter: drop identities with <3 successfully aligned images.
5. Train/val split: 90/10 random within each identity (seeded). Identities themselves NOT held out — Siamese loss benefits from seeing them all during training; we hold out images, not identities, for in-loop sanity. The true held-out evaluation set is LFW.
6. LFW: align all images appearing in `pairs.txt`, save as `lfw_pairs/<name>/<image>.jpg`.
7. Write `manifest.parquet` with columns `(path, identity_id, source, split)`.
8. Notebook `data_processing.ipynb`:
   - Histogram of images per identity (before/after filter).
   - Sample 5×5 grid per dataset.
   - Alignment failure rate per dataset.
   - Total identity count, total image count, split sizes.

### Stage 3 — Train (`training-pipeline/`)

Inputs: `process-data/`.
Outputs: best checkpoint, TensorBoard logs, `training_results.ipynb`.

**Model (`model.py`):**
```python
class FaceEmbedding(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)
```

**Dataset / sampler (`dataset.py`):**
- `PKSampler`: each batch contains P identities × K images (P=32, K=4 → batch=128).
- Standard `torch.utils.data.Dataset` returning `(image_tensor, identity_id)`.
- Augment (train): random horizontal flip, color jitter (mild), random erasing (p=0.2).
- Normalize with ImageNet mean/std.

**Loss (`loss.py`):**
```python
def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    # Pairwise distances
    # For each anchor: hardest positive (max dist same-label) + hardest negative (min dist diff-label)
    # ReLU(d_ap - d_an + margin).mean()
```
Reference: Hermans, Beyer, Leibe — "In Defense of the Triplet Loss for Person Re-identification" (2017).

**Training (`train.py`):**
- Optimizer: `AdamW(params, lr=3e-4, weight_decay=1e-4)`. Use param groups: backbone lr=3e-5, head lr=3e-4.
- Schedule: cosine annealing over total steps. Warmup 500 steps.
- Mixed precision: `torch.cuda.amp.autocast` + `GradScaler`.
- DataLoader: `num_workers=8`, `pin_memory=True`, `persistent_workers=True`.
- Epochs: 20 (configurable). Estimated time at 500K images / batch 128 ≈ 3,900 steps/epoch ≈ 12 min/epoch on A100 → ~4h total.
- Per-epoch eval on a held-out 1000-pair subset of LFW (cosine similarity threshold via 10-fold CV).
- Save best by LFW val accuracy.
- TensorBoard: loss curve, LR schedule, val accuracy, val ROC-AUC, sample triplet distances.

**Auto-stop:** `train.py` wraps main loop in `try/finally`. On exit (success or crash), uploads final checkpoint to `gs://mk8s-sec-057aa9-siamese/checkpoints/` (bucket created once by `infra/create_vm.sh`) and calls `gcloud compute instances stop $HOSTNAME --zone=$ZONE` to halt billing.

**Notebook `training_results.ipynb`:**
- Load TensorBoard event files; render loss + accuracy curves with matplotlib.
- Final hyperparameters table.
- Best epoch / best LFW val accuracy.
- t-SNE of 1000 random validation embeddings (colored by identity).

### Stage 4 — Evaluate (`evaluation/`)

Inputs: best checkpoint, `process-data/lfw_pairs/`.
Outputs: metrics, `evaluation_results.ipynb`.

Steps:
1. Load `pairs.txt` — 6000 pairs in 10 folds (300 positive + 300 negative per fold).
2. Compute embeddings for all referenced images.
3. For each fold:
   - Pick threshold on 9 folds (max accuracy on cosine similarity).
   - Apply threshold to held-out fold.
4. Report: mean accuracy ± std across folds, AUC, ROC curve, threshold @ TPR=99% FPR.
5. Notebook with:
   - ROC curve.
   - Distribution of pos vs neg cosine similarities.
   - Top-10 hardest false positives and false negatives (with images).

### Stage 5 — Application (`application/`)

**Backend (FastAPI):**

Endpoints:
| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/enroll` | `{name: str, image: base64 jpeg}` | `{id, name}` |
| `POST` | `/verify` | `{image: base64 jpeg}` | `{best_match: str\|null, score: float, threshold: float, matched: bool}` |
| `GET` | `/enrolled` | — | `[{id, name, num_samples}]` |
| `DELETE` | `/enrolled/{id}` | — | `{ok: true}` |
| `GET` | `/` | — | serves `index.html` |
| `GET` | `/static/*` | — | serves frontend |

**Storage (`db.py`):**
- SQLite file `embeddings/index.db` with table `enrolled(id, name, npy_path, created_at)`.
- Each enrollment appends a 512-d float32 vector to `embeddings/vectors.npy` (memory-mapped numpy `.npy` extended via concat in-memory then rewrite on enroll/delete — fine for ≤10K entries).
- On startup, load all vectors into a `(N, 512)` torch tensor on CPU.

**Inference (`inference.py`):**
- Load checkpoint once at startup.
- Run on CPU (model is small enough; avoids GPU dependency for demo).
- Preprocess: MTCNN align → 160×160 → normalize → embed → L2-normalize.

**Verify logic:**
- Cosine similarity = matmul against enrolled tensor.
- `best_match` = argmax. `matched` = `best_score >= threshold` (threshold from evaluation stage, e.g., 0.55).

**Frontend (`index.html` + `app.js`):**
- Two tabs/sections: Enroll / Verify.
- Live preview via `<video>` + `getUserMedia({video: true})`.
- Capture button draws current frame to `<canvas>`, exports as base64 JPEG, POSTs to backend.
- Enroll: prompt for name first.
- Verify: shows result card with best match name, score bar, match/no-match indicator.
- Enrolled list with delete buttons.

**Style:** clean light/dark theme via CSS `prefers-color-scheme`; responsive layout; 44px touch targets per global preferences.

**Local run:**
```
cd application && pip install -r requirements.txt && uvicorn backend.main:app --reload
```
Open `http://localhost:8000`.

### GCP Infra (`infra/`)

**`create_vm.sh`:**
```bash
gcloud compute instances create siamese-train \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=common-cu124 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

Fallback if A100 quota denied: change `a2-highgpu-1g` → `g2-standard-8` with `nvidia-l4`.

**`setup_vm.sh`** (runs on VM):
1. `conda activate base` (Deep Learning VM image has PyTorch+CUDA preinstalled).
2. `pip install kaggle facenet-pytorch tensorboard pandas pyarrow fastapi uvicorn`.
3. `mkdir -p ~/.kaggle && echo $KAGGLE_JSON > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json`.
4. `git clone <repo>` + `cd IT4432E_Project`.
5. Download datasets per Stage 1.
6. Run process stage per Stage 2.

**`run_training.sh`:**
- `tmux new -d -s train 'python training-pipeline/src/train.py 2>&1 | tee train.log'`.
- Attach with `tmux attach -t train`.

**Auto-stop guarantee:** done from inside `train.py` rather than shell, so it runs on success, on `Ctrl+C`, and on uncaught exception.

### Secrets

Kaggle token + username never enter git. Stored only:
- On developer laptop in this conversation (already given).
- On VM at `~/.kaggle/kaggle.json` (set by `setup_vm.sh` with `KAGGLE_JSON` env var passed in, never echoed to log).

`.gitignore` excludes:
- `preprocess-data/**` (except `README.md`)
- `process-data/train/`, `process-data/val/`, `process-data/lfw_pairs/`, `process-data/manifest.parquet`
- `training-pipeline/checkpoints/`, `training-pipeline/tensorboard_logs/`
- `application/embeddings/`, `application/models/`
- `**/.kaggle/`, `**/kaggle.json`
- Standard Python: `__pycache__/`, `*.pyc`, `.venv/`, `.ipynb_checkpoints/`

## Data Flow

```
Kaggle ─(API)─▶ /preprocess-data ─(MTCNN align)─▶ /process-data
                                                        │
                                            ┌───────────┴───────────┐
                                            ▼                       ▼
                                   /training-pipeline       /evaluation (LFW)
                                            │                       │
                                            ▼                       ▼
                                  checkpoints/best.pt        metrics report
                                            │
                                            ▼
                                /application (FastAPI + webcam)
```

## Error Handling

- **Kaggle download fails (403/quota):** abort with clear message; user re-checks token.
- **MTCNN finds no face in an image:** log to `process-data/alignment_failures.csv`, skip image. Failure rate reported in notebook.
- **A100 quota denied:** `create_vm.sh` retries on L4 (`g2-standard-8`). Training script tunes batch size based on free VRAM.
- **Out of memory mid-training:** halve batch size, restart. Mixed precision should prevent this on 40GB.
- **VM crashes / preemption:** checkpoints saved every epoch; resume by re-running with `--resume <ckpt>`.
- **App `/verify` with empty enrolled set:** return `{best_match: null, matched: false}`.
- **App face detection fails:** return 422 with message "no face detected".
- **App embedding mismatch on load (dim ≠ 512):** refuse to start, log clear error.

## Testing

Per CLAUDE.md: test behavior, not implementation; tests must call real code; propose failing tests first.

- **Unit (small, fast):**
  - `loss.py`: triplet loss with handcrafted embeddings returns expected value.
  - `db.py`: enroll/verify/delete roundtrip with fake embeddings.
  - `face_align.py`: known face image produces aligned crop of size (160,160,3).
- **Integration:**
  - End-to-end smoke test: load 10 LFW images, enroll 5 unique identities, verify the other 5; assert 4+ correct matches at threshold from evaluation stage.
  - Training script: 2-epoch dry run on tiny synthetic dataset; assert loss decreases.
- **Eval check:** LFW evaluation must hit ≥95% accuracy (FaceNet 99.6%, ResNet50 triplet typically 97-98%). Below 95% → flag and investigate.

## Open Questions

None blocking. Items to revisit during implementation:
- Should we also train on CelebA alone for comparison? — deferred.
- Add liveness detection? — out of scope v1.
- HTTPS / auth for the web app? — out of scope v1.

## Trade-offs

- **CASIA + CelebA combined vs CASIA alone:** combined gives more identities and better alignment (CelebA), at the cost of identity-label collision risk (mitigated by prefix). Worth it.
- **Triplet loss vs ArcFace:** triplet is simpler, well-understood, fits 4-5h budget. ArcFace would yield ~1-2% better LFW accuracy but increases training complexity (margin schedule, classifier head sized to identity count).
- **ResNet50 vs MobileNet:** ResNet50 has better accuracy; inference latency on CPU is ~50-80ms per face, acceptable for demo.
- **CPU inference in app:** trades latency for simplicity. App runs anywhere without GPU. If too slow in practice, switch to GPU inference is a one-line change.
- **A100 vs L4:** A100 hits 4-5h target reliably; L4 is 5x cheaper but risks overshooting budget.

## Success Criteria

- LFW evaluation ≥95% accuracy across 10 folds.
- Training completes within 5 wall-clock hours and ~$22 GCP spend.
- Web app demos full enroll → verify flow end-to-end in a browser on the developer's laptop.
- Repo committed to git in 5 logical stages (one commit per stage minimum).
