# Face Recognition Project - IT4432E

Pins identities are disjoint from both the training data (CASIA) and the eval set (LFW), so this number is a real cross-dataset check — not a re-tuned threshold. Full numbers in `evaluation/results/`.

## Layout

- `shared/preprocess-data/` — raw Kaggle datasets (gitignored) and a sanity-check script
- `shared/process-data/` — MTCNN alignment output (manifest, aligned images)
- `shared/infra/` — VM setup and training launch scripts
- `models/arcface/`, `models/adaface/`, `models/facenet/` — per-model training code
- `evaluation/` — multi-model LFW benchmark CLI, comparison aggregator
- `application/` — FastAPI backend, vanilla-JS webcam frontend, integration tests

## Running the app

Drop trained checkpoints into `application/models/` named `arcface.pt`, `adaface.pt`, or `facenet.pth`.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uvicorn application.backend.main:app --reload
```

Open `http://localhost:8000`, grant camera permission. Three tabs:

- **Enroll** — type a name, capture once, and the person is enrolled into *every* available model at once (each model aligns and embeds the same frame into its own store).
- **Verify** — capture → best match + cosine similarity, run against the model picked in the **Verify model** dropdown.
- **Enrolled** — people grouped across models; each row is tagged with the models it lives in and deletes from all of them in one click.

Threshold defaults are per-model; override with `APP_THRESHOLD_<NAME>` (e.g. `APP_THRESHOLD_ARCFACE=0.6`).

API (CPU only, models loaded once at startup):

- `POST /enroll` — `{name, image}` → `{name, enrolled: [...], failed: [...]}`. Fans out across all available models; `failed` lists any model whose aligner found no face. Errors with 422 only if *no* model could enroll.
- `POST /verify` — `{image, model?, threshold?}` → best match + top candidates for that one model.
- `GET /enrolled` — people grouped by name with the models they appear in. `GET /enrolled?model=<name>` returns that single store's raw entries.
- `DELETE /enrolled/by-name/{name}` — removes a person from every model's store. `DELETE /enrolled/{id}?model=<name>` deletes one entry from one store.

## Tests

```bash
pytest -v
```

Unit tests cover model output shape, triplet loss math, PKSampler invariants, enrollment DB roundtrip, and a short training-loop smoke test. The two integration tests in `application/tests/` require the checkpoint and an LFW sample on disk and auto-skip otherwise.

## Training

Data (from Kaggle):
- [CASIA-WebFace](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) — `debarghamitraroy/casia-webface`. Shipped in InsightFace MXNet RecordIO format; extracted to a folder-per-identity layout before training. Used as the training set.
- [LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) — `jessicali9530/lfw-dataset`. Held out entirely for evaluation (10-fold verification on the canonical 6,000 pairs).

## Models & model selection

Three trained models live side-by-side under `models/`: ArcFace (ResNet50, 160px,
ImageNet norm), AdaFace (IResNet50, 112px, 0.5 norm), FaceNet (InceptionResnetV1,
160px, 0.5 norm). They produce non-comparable embeddings, so each has its own
enrollment store under `application/embeddings/<model>/`. Enrollment writes a
person into every available store at once; verification runs against a single
model picked at runtime, since scores from different models can't be compared.

### Run the app
Drop checkpoints into `application/models/` named `arcface.pt`, `adaface.pt`,
`facenet.pth`. The app loads whatever is present; a model with no
checkpoint is simply skipped (enroll won't write to it, and it shows as
unavailable in the Verify-model dropdown).

```bash
uvicorn application.backend.main:app --port 8000
```

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

