# Face Recognition with Siamese Network

Course project for IT4432E (HUST). Trains a face embedding network with triplet loss, evaluates it on LFW, and serves it through a small web app that does enrollment and verification from the browser webcam.

Pipeline: CASIA-WebFace → MTCNN alignment → ResNet50 + 512-d embedding head trained with batch-hard triplet loss → LFW 10-fold verification benchmark → FastAPI app with SQLite-backed enrollment store.

LFW verification accuracy: **97.48% ± 1.46%** (10-fold CV, 2,378 pairs). Training ran ~17 minutes on a single H100 80GB.

## Layout

- `preprocess-data/` — raw Kaggle datasets (gitignored) and a sanity-check script
- `process-data/` — MTCNN alignment script, output manifest, and an analysis notebook
- `training_pipeline/` — model, loss, sampler, training entry, results notebook, unit tests
- `evaluation/` — LFW 10-fold benchmark CLI, results JSON, report notebook, figures
- `application/` — FastAPI backend, vanilla-JS webcam frontend, integration tests, Dockerfile
- `infra/` — VM setup and training launch scripts
- `docs/superpowers/` — design spec and implementation plan

## Running the app

The trained checkpoint lives at `application/models/best.pt`.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uvicorn application.backend.main:app --reload
```

Open `http://localhost:8000`, grant camera permission. The page has three tabs: Enroll (name + capture), Verify (capture → best match + cosine similarity), Enrolled (list / delete). Threshold defaults to 0.5; override with `APP_THRESHOLD`.

API: `POST /enroll`, `POST /verify`, `GET /enrolled`, `DELETE /enrolled/{id}`. Both `enroll` and `verify` accept `{image: <base64 jpeg>}`; `enroll` additionally takes `name`. The model is loaded once at startup and runs on CPU.

## Tests

```bash
pytest -v
```

Unit tests cover model output shape, triplet loss math, PKSampler invariants, enrollment DB roundtrip, and a short training-loop smoke test. The two integration tests in `application/tests/` require the checkpoint and an LFW sample on disk and auto-skip otherwise.

## Training

Data (from Kaggle):
- [CASIA-WebFace (cropped)](https://www.kaggle.com/datasets/nhatdealin/casiawebface-dataset-crop) — `nhatdealin/casiawebface-dataset-crop`. 1,249 identities, capped at 50 images per identity, 90/10 train/val split → 47,751 aligned faces for training.
- [LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) — `jessicali9530/lfw-dataset`. 901 identities, capped at 200, 7,240 aligned faces, held out entirely for evaluation.
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) — `jessicali9530/celeba-dataset`. Was in the original plan but this Kaggle dump shipped without `identity_CelebA.txt`, so it was dropped.

Reference implementation that shaped the design: [Face Recognition with Siamese Network](https://www.kaggle.com/code/tatianakushniruk/face-recognition-with-siamese-network) by tatianakushniruk.

Model: `torchvision.models.resnet50` (ImageNet V2 weights) feeds a `Linear(2048, 512)` head whose output is L2-normalised. Loss is batch-hard triplet (margin 0.3, Hermans et al. 2017). Batches are built by a PKSampler — 32 identities × 4 images = 128 samples per batch, 1,500 batches per epoch, 20 epochs. AdamW with split learning rates (backbone 3e-5, head 3e-4), cosine schedule with 500-step warmup, mixed precision.

Final train loss 0.14 (down from 0.33). Best LFW in-loop accuracy 0.976 on a 1k-pair subset; full 10-fold CV reported above.

Reproducing training:
```bash
bash infra/setup_vm.sh         # installs PyTorch + downloads + aligns datasets
bash infra/run_training.sh     # 20 epochs, ~17 min on H100
python -m evaluation.eval_lfw  # 10-fold benchmark, writes evaluation/results.json
```

## Notes from the build

- `training-pipeline/` became `training_pipeline/` so Python's import system would accept it.
- Original infra plan targeted GCP A100. The project's GCP account had global GPU quota fixed at zero and was ineligible for an increase, so training moved to a non-GCP H100 provider. `infra/create_vm.sh` is the GCP path and is unused; `infra/setup_vm.sh` is the generic Ubuntu-22.04-with-NVIDIA-driver path that actually ran.
- The provider has no auto-stop API. The VM must be shut down manually from the provider dashboard after `best.pt` is pulled, or it will keep billing.
