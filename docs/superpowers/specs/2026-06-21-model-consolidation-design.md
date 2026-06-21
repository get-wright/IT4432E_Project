# Consolidate Face-Recognition Models into `main`

**Date:** 2026-06-21
**Status:** Approved design, pending implementation plan

## Goal

Bring three independently-trained face-recognition models onto `main` under one
organized structure, give the application a runtime model selector, and
standardize evaluation so every model reports **Accuracy, Precision, Recall,
F1-Score** (plus ROC-AUC) from one shared evaluator.

## Source branches

| Branch | Model | Pipeline today | App today | Eval metrics today |
|---|---|---|---|---|
| `feat/arcface-tta` | ResNet50 + ArcFace + TTA, 160×160 | full (preprocess→train→eval) | FastAPI | **Accuracy only** (LFW mean_acc 0.909) + sim stats |
| `thanh/feat/adaface` | IResNet50 + AdaFace, 112×112 | full | FastAPI | **Full** — 8 InsightFace suites, Acc/Prec/Recall/F1/ROC-AUC |
| `Facenet-Model` | InceptionResnetV1 (vggface2) + triplet, 160×160 | **notebook only** (`casia.ipynb`) | none | Accuracy + ROC-AUC (no Prec/Recall/F1) |

`feat/arcface-tta` and `thanh/feat/adaface` are near-identical forks of a shared
siamese base; their `preprocess-data/`, `process-data/`, `infra/`, app skeleton,
and most of `training_pipeline/` match. `Facenet-Model` is a single notebook plus
a checkpoint.

## Key facts established during exploration

- **Weights are not in git** (gitignored). On disk: only `facenet_best.pth`
  (`/Users/n3m0/Downloads/`, key format `{'model_state_dict', 'epoch', 'best_acc'}`,
  state-dict keys prefixed `bb.`). ArcFace and AdaFace checkpoints are **not
  present anywhere locally** — `application/models/` is empty on every copy checked.
- **No torch / no GPU / no eval data** in the consolidation environment. Therefore
  the evaluator is a **code deliverable**; the user runs it where weights + LFW
  data + torch exist.
- **Metrics are recreatable without retraining.** Precision/Recall/F1 are pure
  functions of (cosine sims, labels, threshold). All three evaluators already
  produce sims + a tuned threshold; AdaFace's code already computes the full set.
  Lifting that into one shared function gives ArcFace and FaceNet identical metrics.
- **Embeddings are not comparable across models** (different backbones, different
  input sizes 160 vs 112, all 512-d but distinct spaces). The app must keep a
  **separate enrollment store per model**.
- **Per-model inference contracts differ and must NOT be unified:**
  | Model | Input size | Normalization | Default threshold | TTA |
  |---|---|---|---|---|
  | arcface | 160 | **ImageNet** (mean `[.485,.456,.406]` std `[.229,.224,.225]`) | 0.565 | flag, default **off** (LFW acc 0.909 no-TTA vs 0.847 TTA) |
  | adaface | 112 | 0.5 / 0.5 | 0.5 (LFW CV thr 0.237) | n/a |
  | facenet | 160 | 0.5 / 0.5 | from eval (ROC-optimal) | n/a |

  Confirmed in source: ArcFace app aligner + `dataset.py` use ImageNet norm;
  AdaFace app aligner uses 0.5/0.5. A single global normalization would
  **silently regress ArcFace** — normalization is per-model registry metadata.
- ArcFace's app already has an **`embedding_version` guard** (`db.py`): the store
  records a checkpoint-hash + TTA id and refuses to load enrollments made under a
  different model version. This must be **preserved** — per-model store folders
  alone do not catch "same model name, new weights".

## Decisions (locked)

1. **Side-by-side organization** — keep each model's training code; add shared
   evaluator + multi-model app on top. No risky merge of diverged pipelines.
2. **Runtime model switch in the UI** — dropdown; backend loads all available
   checkpoints, routes enroll/verify by selected model, per-model stores.
3. **Wire FaceNet into the app** — add a FaceNet wrapper so it is selectable.
4. **Top layout `models/<name>/` + `shared/`.**
5. **Weights gitignored + documented** drop-zone (no LFS).
6. **Metrics: code-only deliverable** — shared evaluator + comparison table; user
   runs it later with weights/data.
7. **ArcFace TTA = registry flag, default off** (no-TTA scored higher on LFW).
8. **Benchmark scope: LFW common across all 3** for the comparison table; AdaFace's
   8-suite InsightFace report preserved as an AdaFace-only extended artifact.
9. **No global norm/size/threshold** — all per-model in the registry; preserve the
   `embedding_version` enrollment guard for every model.

## Target structure on `main`

```
IT4432E_Project/
├── models/
│   ├── arcface/          # ResNet50+ArcFace: model.py, arcface_head.py, train.py, configs
│   ├── adaface/          # IResNet50+AdaFace: iresnet.py, train.py, configs
│   └── facenet/          # casia.ipynb + facenet_model.py (InceptionResnetV1 wrapper)
├── application/
│   ├── backend/
│   │   ├── registry.py   # name → (builder, ckpt path, input size, ckpt-key, norm)
│   │   ├── inference.py  # Embedder dispatches by registry entry
│   │   ├── main.py       # /models endpoint + model field on enroll/verify
│   │   ├── face_align.py # aligns at the selected model's input size
│   │   └── db.py         # per-model embedding stores
│   ├── frontend/         # adds model dropdown populated from GET /models
│   └── models/           # gitignored weights drop-zone: arcface.pt, adaface.pt, facenet.pth
├── evaluation/
│   ├── evaluate.py       # shared: any model → Acc/Prec/Recall/F1/ROC-AUC + confusion
│   ├── metrics.py        # pure fn(sims, labels, threshold) → metric dict
│   └── results/          # per-model JSON + combined comparison table (md + csv)
├── shared/
│   ├── preprocess-data/
│   ├── process-data/
│   └── infra/
├── docs/
└── README.md
```

Shared pipeline stages live once in `shared/`. Each `models/<name>/` keeps only
what differs (backbone, loss, train config). The notebook stays as the FaceNet
training artifact; a thin `facenet_model.py` wrapper is added for app + eval.

## Components

### 1. Model registry (`application/backend/registry.py`)
Single source of truth mapping model name → spec. Every field that the three
models disagree on lives here — there is no global default for norm/size/threshold:
```
{
  "arcface": {builder, ckpt: "arcface.pt", input_size: 160,
              norm: IMAGENET,  ckpt_key: "model", needs_cfg: true,
              threshold: 0.565, use_tta: false},
  "adaface": {builder, ckpt: "adaface.pt", input_size: 112,
              norm: HALF,      ckpt_key: "model", needs_cfg: false,
              threshold: 0.5},
  "facenet": {builder, ckpt: "facenet.pth", input_size: 160,
              norm: HALF,      ckpt_key: "model_state_dict",
              threshold: <from eval>},
}
```
- `norm` is `(mean, std)` per model — `IMAGENET` for arcface, `HALF`=([.5]*3,[.5]*3)
  for adaface/facenet. The aligner reads norm + input_size from the registry.
- `threshold` is the per-model default; env override is **scoped per model**
  (`APP_THRESHOLD_ARCFACE`, …), never a single global `APP_THRESHOLD`. Defaults
  may be sourced from each model's `results/<model>.json` tuned threshold.
- `use_tta` is an arcface-only flag, default **off**. The app exposes it; the
  registry's `embedding_version` (below) incorporates it so toggling TTA
  invalidates stale enrollments.

Each builder returns a module with `embed_normalized(x) -> unit-norm 512-d`.
FaceNet builder mirrors the notebook's `FaceNet` wrapper — it holds the backbone
as `self.bb = InceptionResnetV1(pretrained=None, classify=False)` so the
checkpoint's `bb.`-prefixed `model_state_dict` keys load directly with no prefix
stripping; `embed_normalized` applies L2-norm to `bb(x)`. Load with
**`strict=True`** so a shape/prefix mismatch fails loudly rather than silently
loading a partial model.

### 1a. Enrollment versioning (preserve ArcFace's guard for all models)
Carry over ArcFace's `embedding_version` mechanism (`db.py`) to every model: each
per-model store records a version = `sha256(checkpoint)[:16]` + model name +
(arcface) TTA flag. On load, a mismatch raises a clear "re-enroll / clear store"
error. This guards against replacing a checkpoint or flipping TTA under the same
model name — per-model folders are necessary but not sufficient.

### 2. Multi-model app
- Startup: instantiate every model whose checkpoint exists in
  `application/models/`. A missing checkpoint marks that model `available: false`
  (not a crash, not omitted).
- `GET /models` → returns **all registry entries**:
  `[{name, available, input_size, reason?}]`. The frontend needs unavailable
  entries to grey them out; `reason` explains the gap (e.g. "checkpoint missing").
- `POST /enroll` / `POST /verify` accept `model` field; route to that model's
  embedder + its own store under `embeddings/<model>/`, using that model's
  registry `threshold` (per-model env override). Default model = first available.
- Aligner runs at the selected model's `input_size` **and** normalization.
- Selecting an `available: false` model → 503.
- Frontend: dropdown from `GET /models`; available models selectable, unavailable
  greyed out with `reason`. Switching changes the active store.
- **Security:** app stays unauthenticated (local demo). README flags this; auth
  is out of scope unless requested.

### 3. Shared evaluator (`evaluation/metrics.py` + `evaluate.py`)
- `metrics.py`: pure `compute_metrics(sims, labels, threshold)` →
  **canonical schema** `{accuracy, precision, recall, f1, roc_auc, threshold, tp,
  fp, tn, fn}`. All models emit this exact key set — no `mean_acc` vs
  `accuracy_cv` vs `mean_acc_cv` mixing. Confusion metrics (P/R/F1) are computed
  at the **median 10-fold CV threshold** (the value reported as `threshold`), so
  every model's P/R/F1 is at a comparable operating point. `accuracy` is the
  held-out 10-fold CV mean; `std_acc` is also recorded.
- **Preprocessing parity:** `evaluate.py` embeds each model's pairs using the
  **same `input_size` + normalization + alignment path as the app registry** for
  that model (arcface→160/ImageNet, adaface→112/0.5, facenet→160/0.5). The
  evaluator must not hardcode ArcFace's ImageNet/160 loading for all models; it
  pulls preprocessing from the registry so eval and inference agree.
- `evaluate.py`: `--model <name> --pairs ... --data ...` → loads the registry
  model, embeds pairs at that model's contract, 10-fold CV threshold, writes
  `results/<model>.json` in the canonical schema.
- **Benchmark scope:** all three models are evaluated on **LFW** for the
  apples-to-apples `results/comparison.{md,csv}` table (LFW is the common
  denominator all three can run). AdaFace's existing **8-suite InsightFace report**
  is preserved as its own extended artifact, clearly labeled as AdaFace-only and
  not mixed into the 3-model comparison.
- Backfills Prec/Recall/F1 for ArcFace and FaceNet with **no retraining**.

## Data flow

Enroll: image → align(model.input_size, model.norm) → embedder(model) → store in
`embeddings/<model>/` (tagged with model's embedding_version).
Verify: image → align → embed → cosine vs that model's store → model.threshold → match.
Eval: pairs → align/embed at **model's registry contract** → sims+labels →
`compute_metrics(sims, labels, cv_threshold)` → `results/<model>.json` → comparison table.

## Error handling

- Missing checkpoint → model excluded from `GET /models`; selecting it → 503.
- No face detected → 422 (existing behavior preserved).
- Verify against empty store → "no enrollments" response, not a crash.
- Evaluator missing weights/data → clear error naming the missing path.

## Testing

- Registry: each entry builds and loads its checkpoint format with `strict=True`
  (mock/sample state dict); norm + input_size + threshold present per model.
- App: enroll/verify round-trip per available model; `/models` returns all entries
  with correct `available`; cross-model isolation (enrolling in A doesn't leak
  into B); `embedding_version` mismatch raises on a tampered store; selecting an
  unavailable model → 503.
- `metrics.py`: one self-check with a synthetic sims/labels array where
  Acc/Prec/Recall/F1 are known by hand (the required runnable check for the money path).
- Eval preprocessing parity: assert the evaluator's per-model transform matches
  the registry (size + norm), so eval and app inference can't silently diverge.
- Preserve existing app + pipeline tests carried over from the branches.

## What this delivers vs. what the user runs

**Delivered in `main`:** organized tree, multi-model selectable app, FaceNet
wrapper, shared evaluator + comparison table, gitignored weights drop-zone +
README instructions.

**User runs later (needs weights + LFW data + torch/GPU):** `evaluate.py` per
model to populate `results/` and the comparison table; drops the three
checkpoints into `application/models/` to use the app. ArcFace and AdaFace
checkpoints must be retrieved/regenerated by the user (not on disk); FaceNet's is
already at `/Users/n3m0/Downloads/facenet_best.pth`.

## Out of scope

- Retraining any model.
- Unifying the three training pipelines into one config-driven pipeline.
- Authentication / deployment hardening.
- Git LFS for weights.
