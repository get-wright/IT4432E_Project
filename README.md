# Face Recognition with Siamese Network

ResNet50 + triplet-loss Siamese network trained on CASIA-WebFace, evaluated on LFW, served via a FastAPI web app that captures faces from the browser webcam.

**Result: 97.48% ± 1.46% LFW verification accuracy** (10-fold cross-validation, 2378 pairs). See [`evaluation/results.json`](evaluation/results.json) and [`evaluation/evaluation_results.ipynb`](evaluation/evaluation_results.ipynb).

See [design spec](docs/superpowers/specs/2026-05-17-face-recognition-siamese-design.md) and [implementation plan](docs/superpowers/plans/2026-05-17-face-recognition-siamese.md) for full context.

## Repo layout
- `preprocess-data/` — raw datasets (gitignored) + sanity-check script
- `process-data/` — MTCNN-aligned faces + analysis notebook
- `training_pipeline/` — model, loss, training entry, results notebook
- `evaluation/` — final LFW benchmark + report notebook + figures
- `application/` — FastAPI backend + browser-webcam frontend
- `infra/` — GPU VM provisioning + dataset download scripts

## Quickstart

After cloning, with a trained `application/models/best.pt` checkpoint in place:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cd application
APP_THRESHOLD=0.5 uvicorn backend.main:app --reload
```

Open <http://localhost:8000>, grant camera permission, enroll a face, then verify.

## Tests

```bash
pytest -v
```

9 unit tests (model shapes, triplet loss math, sampler invariants, DB roundtrip, training-loop smoke test). The 2 integration tests in `application/tests/` auto-skip if no checkpoint / no LFW sample is present locally.

## Training pipeline

Datasets:
- **CASIA-WebFace** (raw images from `nhatdealin/casiawebface-dataset-crop`) — 1,249 identities × up to 50 images = 47,751 aligned faces (90/10 train/val split).
- **LFW** (`jessicali9530/lfw-dataset`) — 901 identities × up to 200 images = 7,240 aligned faces. Used only for evaluation (held out from training).
- **CelebA** was attempted but the Kaggle dump lacked `identity_CelebA.txt` (only attribute CSVs) — dropped.

Architecture:
- `torchvision.models.resnet50` (ImageNet V2 pretrained) → Linear(2048, 512) → L2 normalize.
- **BatchHard triplet loss** with margin 0.3 (Hermans et al., 2017).
- **PKSampler** — each batch is 32 identities × 4 images = 128 samples.
- AdamW with separate LRs: backbone 3e-5, head 3e-4. Cosine schedule with 500-step warmup. Mixed precision.
- 20 epochs × 1500 batches = 30K steps. Wall-clock: ~17 minutes on a single NVIDIA H100 80GB.

Final training loss 0.14 (from 0.33). Best LFW in-loop accuracy 0.976 (1000-pair subset, threshold tuned via 10-fold CV).

## Application

- `POST /enroll` — accepts `{name, image: base64 jpeg}`, runs MTCNN alignment, computes embedding, stores in SQLite + `.npy`.
- `POST /verify` — accepts `{image: base64 jpeg}`, computes embedding, cosine similarity vs all enrolled, returns best match + score + threshold + matched flag.
- `GET /enrolled` / `DELETE /enrolled/{id}` — list/remove enrollments.
- Frontend is a single page (vanilla HTML + JS + CSS) with live webcam preview and three tabs (Enroll / Verify / Enrolled).
- Light + dark themes via `prefers-color-scheme`. 44px touch targets. Visible focus outlines.

## Infra

Training ran on a Vietnamese provider 1×H100 80GB VM (~$2.70/hr × ~30min ≈ $1.50). See `infra/setup_vm.sh` for the VM-side install + dataset download script. Original `infra/create_vm.sh` targets GCP A100 (replace if you're not on GCP).

## Adjusted from original design spec

- Directory `training-pipeline/` renamed to `training_pipeline/` to satisfy Python import system (hyphens aren't valid in module names).
- CelebA dropped — the Kaggle dump lacked identity labels.
- GCP A100 replaced with a non-GCP H100 (project had zero `GPUS_ALL_REGIONS` quota and Google denied the increase).
- `utils.py` doesn't ship `upload_to_gcs` / `stop_vm_self` — handled out-of-band on the non-GCP provider.
- `infra/create_vm.sh` is now provider-flavoured for GCP only and is not used in the final training run; `infra/setup_vm.sh` was generalised for Ubuntu 22.04 + NVIDIA-driver VMs.

## Cost guardrail

The H100 VM is **not** automatically stopped after training. Manually shut it down via your provider's dashboard once you've pulled `application/models/best.pt` to your laptop.
