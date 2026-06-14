# Face Recognition with IResNet50 + AdaFace

Course project for IT4432E (HUST). Trains an **IResNet50** face embedding network with the **AdaFace** adaptive-margin loss, evaluates it on 8 InsightFace verification benchmarks, and serves it through a FastAPI web app with webcam-based enroll/verify.

Pipeline: CASIA-WebFace → IResNet50 + AdaFace loss (30 epochs, RTX 4070) → 8-benchmark InsightFace suite → FastAPI app with SQLite-backed enrollment store.

---

## Results

### LFW (primary benchmark)

| Metric | Value |
|---|---|
| Mean accuracy (10-fold CV, 6,000 pairs) | **99.30% ± 0.40%** |
| F1 score | **0.9938** |
| Precision | 0.9963 |
| Recall | 0.9913 |
| ROC AUC | **0.9995** |
| TPR @ FPR = 1% | see `results_adaface.json` |
| Spread (pos\_sim − neg\_sim) | 0.5815 |
| Threshold τ (cosine similarity) | 0.2370 |

> All metrics are from 10-fold CV on the InsightFace LFW `.bin` (pre-aligned 112×112, AdaFace normalization).  
> Re-run `evaluation/adaface_evaluation.ipynb` to regenerate from `application/models/best.pt`.

### InsightFace 8-benchmark suite

| Benchmark | Description | Acc (CV-tuned τ) | Std | F1 | ROC-AUC | Spread |
|---|---|---|---|---|---|---|
| **lfw** | Standard frontal verification | **99.30%** | ±0.40% | 0.9938 | 0.9995 | 0.582 |
| **cfp\_ff** | Frontal vs frontal | **99.47%** | ±0.26% | 0.9949 | 0.9996 | 0.615 |
| **cfp\_fp** | Frontal vs profile (±90°) | **95.03%** | ±1.08% | 0.9491 | 0.9766 | 0.395 |
| **agedb\_30** | Same person, 30-year age gap | **94.25%** | ±1.26% | 0.9443 | 0.9831 | 0.345 |
| **calfw** | Cross-age LFW | **93.48%** | ±0.97% | 0.9334 | 0.9732 | 0.415 |
| **cplfw** | Cross-pose LFW (extreme yaw) | **89.28%** | ±1.60% | 0.8879 | 0.9388 | 0.319 |
| **sllfw** | Similar-looking lookalikes | **98.05%** | ±0.60% | 0.9815 | 0.9962 | 0.478 |
| **talfw** | Transfer-attack LFW (adversarial) | 50.00% | ±0.00% | 0.667 | 0.417 | −0.073 |

Full per-benchmark numbers with precision, recall, TP/FP/TN/FN, confusion matrices, and per-fold breakdowns are in `evaluation/adaface_evaluation.ipynb` and `evaluation/results_adaface.json`.

**Reading the results:**

- **LFW / CFP-FF (99.3% / 99.5%)** — Excellent performance on clean frontal pairs. AdaFace's norm-aware margin produces tightly-clustered, well-separated embeddings (spread ≈ 0.58–0.62).
- **CFP-FP (95.0%)** — Only −4.3 pp drop for profile faces. The IResNet50 backbone handles moderate pose variation well.
- **AgeDB-30 / CALFW (94.3% / 93.5%)** — Strong age robustness, significantly better than a standard triplet-trained model. AdaFace's adaptive margin up-weights low-norm (hard/occluded) samples, which reduces the model's reliance on age-correlated texture features.
- **CPLFW (89.3%)** — Extreme-yaw pairs cause the largest in-scope drop (−10 pp vs LFW). Profile images are under-represented in CASIA-WebFace.
- **SLLFW (98.1%)** — Near-LFW performance on lookalikes; the adaptive margin provides meaningful hard-negative separation without explicit mining.
- **TALFW (50.0%, AUC 0.42)** — Complete failure on adversarial perturbations. The negative AUC (< 0.5) means perturbations *invert* the similarity ordering — adversarially crafted images appear more similar to wrong identities than to the correct one. This is a known property of face recognition models without adversarial training and is out of scope for this project.

---

## Model

**Architecture:** IResNet50 — the InsightFace ResNet-50 variant with pre-activation `IBasicBlock` residual blocks, PReLU activations, and a `Linear(25088, 512)` + `BatchNorm1d(512)` embedding head (no global-average-pool; spatial features flattened directly).

**Loss:** AdaFace (Kim et al., 2022). Adaptive angular margin: samples with high feature-norm (confident, well-lit faces) receive a larger margin penalty; low-norm samples (blurry, occluded) get a smaller penalty. This implicitly up-weights hard examples without explicit hard-mining.

**Embedding:** 512-d, L2-normalised to the unit hypersphere at inference time. Verification score = cosine similarity between two embeddings.

---

## Training

### Data

| Dataset | Source | Role |
|---|---|---|
| CASIA-WebFace | `debarghamitraroy/casia-webface` (Kaggle) | Training (10,572 identities, ≈490K images) |
| LFW | `jessicali9530/lfw-dataset` (Kaggle) | Evaluation only (6,000 canonical pairs) |
| InsightFace eval suite | `data/casia-webface/eval/*.bin` | Evaluation only (8 benchmarks) |

CASIA-WebFace is shipped as InsightFace MXNet RecordIO (`.rec`/`.idx`/`property`). `train_local.py` reads it with a pure-Python RecordIO parser (no MXNet dependency).

### Hyperparameters

| Parameter | Value |
|---|---|
| Architecture | IResNet50 (layers: 3-4-14-3) |
| Loss | AdaFace, m=0.4, h=0.333, s=64.0, t\_alpha=0.01 |
| Embedding size | 512 |
| Input size | 112 × 112 |
| Input normalisation | mean = std = 0.5 (range [-1, 1]) |
| Epochs | 30 |
| Batch size | 256 |
| Optimiser | SGD, momentum=0.9, weight\_decay=5e-4 |
| Learning rate | 0.05 |
| LR schedule | Warmup 1 epoch → MultiStep ×0.1 at epochs 16, 24, 28 |
| Gradient clipping | max\_norm=5.0 |
| Mixed precision | FP16 (CUDA AMP) |
| Hardware | NVIDIA RTX 4070 12 GB |

### Training script

```bash
# Run from the repo root (adaface/), not IT4432E_Project/
python train_local.py
```

The script saves `ckpt/last.pt` (full training state: model, head, optimizer, scaler, epoch) after every epoch and `ckpt/embedder.pt` (model weights only) at the end of training. Copy `embedder.pt` to `IT4432E_Project/application/models/best.pt` to deploy.

---

## Layout

- `preprocess-data/` — raw Kaggle datasets (gitignored) and a sanity-check script
- `process-data/` — face alignment scripts, output manifest, analysis notebook
- `training_pipeline/` — original two-phase CE+triplet pipeline (coursework scaffold)
- `evaluation/` — 8-benchmark evaluation notebook, results JSON, figures
- `application/` — FastAPI backend, vanilla-JS webcam frontend, integration tests, Dockerfile
- `infra/` — VM setup and training launch scripts
- `docs/` — design spec, training overview

---

## Running the app

The trained checkpoint lives at `application/models/best.pt` (gitignored). Download the
pre-trained weights (≈167 MB) from Google Drive:

- **Download:** https://drive.google.com/file/d/1SMPbpQ60rQEJeNORMB5q4jn_7u6zU8-I/view?usp=sharing

Place the downloaded file at `application/models/best.pt`. From the command line:

```bash
cd IT4432E_Project
mkdir -p application/models
# requires: pip install gdown
gdown 1SMPbpQ60rQEJeNORMB5q4jn_7u6zU8-I -O application/models/best.pt
```

(Alternatively, if you trained the model yourself, copy `ckpt/embedder.pt` to
`application/models/best.pt`.)

Then set up the environment and launch:

```bash
cd IT4432E_Project
uv venv --python 3.11 .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
uvicorn application.backend.main:app --reload
```

> No `uv`? Use the stdlib instead: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.

Open `http://localhost:8000` (the FastAPI backend serves both the API and the web
frontend on this single port — there is no separate frontend server). Three tabs:

- **Enroll** — enter a name, capture a face from the webcam, store the embedding.
- **Verify** — capture a face, find the closest enrolled identity and return the cosine similarity score.
- **Enrolled** — list all enrolled faces; delete by ID.

Environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `APP_CKPT` | `application/models/best.pt` | Path to model checkpoint |
| `APP_DATA_DIR` | `application/embeddings/` | Enrollment store directory |
| `APP_THRESHOLD` | `0.5` | Cosine similarity accept threshold |

The model is loaded once at startup on CPU. Pass `device="cuda"` in `Embedder.__init__` and update `FaceAligner` if you need GPU inference.

---

## Running evaluation

```bash
cd IT4432E_Project
# Open and run all cells:
jupyter notebook evaluation/adaface_evaluation.ipynb
```

This loads `application/models/best.pt`, reads the 8 InsightFace `.bin` files, runs 10-fold CV on every benchmark, and writes `evaluation/results_adaface.json` with full per-benchmark metrics.

---

## Tests

```bash
cd IT4432E_Project
pytest -v
```

Unit tests cover model output shape, triplet loss math, PKSampler invariants, and DB roundtrip. Integration tests (`application/tests/`) require the checkpoint and an LFW sample on disk; they auto-skip otherwise.

---

## Notes

- `train_local.py` uses a pure-Python RecordIO reader to avoid the MXNet dependency — see `_RecordHandle` / `_unpack`. Works with the Kaggle CASIA-WebFace dump as-is.
- `ckpt/last.pt` contains the full training state and can resume training (it checks for its own existence at startup). `ckpt/embedder.pt` strips the classifier head and optimiser state — use this for deployment.
- The two-phase CE+triplet pipeline (`training_pipeline/`) is the original coursework scaffold. It is retained for reference but `train_local.py` + AdaFace is the main training path used to produce the deployed model.
- Original infra plan targeted GCP A100. Actual training ran on a local RTX 4070. `infra/` scripts are the generic Ubuntu-22.04-with-NVIDIA path.
