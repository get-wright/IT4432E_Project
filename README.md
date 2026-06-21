# Face Recognition with Siamese Network

Course project for IT4432E (HUST). Trains a face embedding network with a two-phase recipe (softmax warmup → semi-hard triplet), evaluates it on LFW, and serves it through a small web app that does enrollment and verification from the browser webcam.

Pipeline: CASIA-WebFace → MTCNN alignment → ResNet50 + 512-d embedding head trained with a two-phase recipe (softmax warmup → semi-hard triplet) → LFW 10-fold verification benchmark → FastAPI app with SQLite-backed enrollment store.

**Results**

| Benchmark | Metric | Value |
|---|---|---|
| LFW (10-fold CV, 6000 pairs, strict) | `mean_acc ± std` | **90.90% ± 1.09%** |
| LFW | `pos_sim − neg_sim` (spread) | 0.585 |
| LFW | threshold (cosine sim) | 0.565 |
| Pins (105 celebs, disjoint from CASIA + LFW) | accuracy at fixed LFW threshold | **77.7%** |
| Pins | spread | 0.376 |

Pins identities are disjoint from both the training data (CASIA) and the eval set (LFW), so this number is a real cross-dataset check — not a re-tuned threshold. Full numbers in `evaluation/results.json`.

For a deeper walk-through of why the training is structured this way — including the bug it replaced — see [`docs/training-overview.md`](docs/training-overview.md).

## Layout

- `shared/preprocess-data/` — raw Kaggle datasets (gitignored) and a sanity-check script
- `shared/process-data/` — MTCNN alignment output (manifest, aligned images)
- `shared/infra/` — VM setup and training launch scripts
- `models/arcface/`, `models/adaface/`, `models/facenet/` — per-model training code
- `evaluation/` — multi-model LFW benchmark CLI, comparison aggregator
- `application/` — FastAPI backend, vanilla-JS webcam frontend, integration tests
- `docs/` — training overview and cross-dataset eval notes

## Running the app

Drop trained checkpoints into `application/models/` named `arcface.pt`, `adaface.pt`, or `facenet.pth`.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uvicorn application.backend.main:app --reload
```

Open `http://localhost:8000`, grant camera permission. The page has three tabs: Enroll (name + capture), Verify (capture → best match + cosine similarity), Enrolled (list / delete). Threshold defaults are per-model; override with `APP_THRESHOLD_<NAME>` (e.g. `APP_THRESHOLD_ARCFACE=0.6`).

API: `POST /enroll`, `POST /verify`, `GET /enrolled`, `DELETE /enrolled/{id}`. Both `enroll` and `verify` accept `{image: <base64 jpeg>}`; `enroll` additionally takes `name`. The model is loaded once at startup and runs on CPU.

## Tests

```bash
pytest -v
```

Unit tests cover model output shape, triplet loss math, PKSampler invariants, enrollment DB roundtrip, and a short training-loop smoke test. The two integration tests in `application/tests/` require the checkpoint and an LFW sample on disk and auto-skip otherwise.

## Training

Data (from Kaggle):
- [CASIA-WebFace](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) — `debarghamitraroy/casia-webface`. Shipped in InsightFace MXNet RecordIO format; extracted to a folder-per-identity layout before training. Used as the training set.
- [LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) — `jessicali9530/lfw-dataset`. Held out entirely for evaluation (10-fold verification on the canonical 6,000 pairs).
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) — `jessicali9530/celeba-dataset`. Was in the original plan but this Kaggle dump shipped without `identity_CelebA.txt`, so it was dropped.

Reference implementation that shaped the early design: [Face Recognition with Siamese Network](https://www.kaggle.com/code/tatianakushniruk/face-recognition-with-siamese-network) by tatianakushniruk.

Model: `torchvision.models.resnet50` (ImageNet V2 weights) feeds a `Linear(2048, 512)` head. `forward()` returns the raw 512-d vector; L2 normalisation happens inside the triplet loss and at inference time via `embed_normalized()`. Training is two-phase:

1. **Softmax warmup (3 epochs).** A temporary `Linear(512, n_identities)` classifier on top of the embedding; cross-entropy on identity labels with `RandomSampler(replacement=True)` and 1,500 batches × 128 images per epoch.
2. **Semi-hard triplet (17 epochs).** PKSampler (P=32 identities × K=4 images = 128 batch), soft-margin triplet loss `softplus(d_ap - d_an)` with `d_ap < d_an < d_ap + margin` mining (margin 0.3, FaceNet 2015).

AdamW with split learning rates (backbone 1e-4, head 5e-4), cosine schedule with 500-step warmup, mixed precision on H100. The end of Phase 1 enforces `spread > 0.05` on an LFW probe — if embeddings collapsed, training aborts before triplet starts. `best.pt` is only promoted when validation accuracy improves *and* spread stays > 0.05.

Reproducing training:
```bash
bash shared/infra/setup_vm.sh         # installs PyTorch + downloads datasets
bash shared/infra/run_training.sh     # ~25 min on H100 (3 warmup + 17 triplet epochs)
python -m evaluation.evaluate --model arcface --pairs <pairs.txt> --aligned-root <dir>
```

## Notes from the build

- `training-pipeline/` became `training_pipeline/` so Python's import system would accept it.
- Original infra plan targeted GCP A100. The project's GCP account had global GPU quota fixed at zero and was ineligible for an increase, so training moved to a non-GCP H100 provider. `infra/create_vm.sh` is the GCP path and is unused; `infra/setup_vm.sh` is the generic Ubuntu-22.04-with-NVIDIA-driver path that actually ran.
- The provider has no auto-stop API. The VM must be shut down manually from the provider dashboard after `best.pt` is pulled, or it will keep billing.

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
