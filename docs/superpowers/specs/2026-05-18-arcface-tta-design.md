# ArcFace + TTA Recipe Upgrade — Design

**Date:** 2026-05-18
**Branch:** `feat/arcface-tta` (off `feat/face-recognition-siamese` @ `cad9788`)
**Goal:** Replace semi-hard triplet loss with ArcFace angular margin loss and add horizontal-flip TTA at inference, targeting +5–10pp on age and cross-pose benchmarks while preserving every existing methodology.

## Why

Cross-dataset benchmarks (commit `cad9788`) show the current Siamese model is solid on in-distribution frontals (LFW 95.03%, CFP-FF 94.79%) but loses 17–19pp on age (AgeDB-30, CALFW) and cross-pose (CPLFW), and 17pp on lookalikes (SLLFW). Soft-margin semi-hard triplet provides only mild pressure on near-boundary cases — once positives are inside the margin, loss → 0. ArcFace's angular margin pushes every class boundary by a fixed angle, producing tighter clusters and better hard-negative separation. Literature shows 99.53% LFW achievable on the same CASIA-WebFace + ResNet50 setup.

## What stays the same

This is a loss-and-inference change, not a methodology change. Untouched:

- Siamese inference architecture (one shared embedding tower, cosine compare)
- Two-phase training shape (CE softmax warmup → margin loss)
- PKSampler P=32 × K=4 = 128 batch
- Collapse gate (`spread > 0.05` on held-out batch)
- best.pt promotion guard (`acc > best_acc AND spread > 0.05`)
- AMP unchanged (existing GradScaler API in `train.py`)
- L2-norm inside loss (model exposes `embed_normalized`)
- MTCNN + raw fallback for eval
- Strict 10-fold LFW protocol (`pos_ratio = 0.500`)
- 8-suite InsightFace .bin benchmark notebook
- App backend (`application/backend/main.py`)

## Architecture

### Branch + artifacts

`feat/arcface-tta` is created off `feat/face-recognition-siamese` HEAD. Existing `train.yaml`, `best.pt`, and all current eval result JSONs stay in place. New artifacts:

- `training_pipeline/checkpoints/best_arcface.pt` — embedding tower trained with ArcFace
- `training_pipeline/checkpoints/arcface_head.pt` — head weights (saved for analysis only, never loaded by app)
- `evaluation/results_arcface.json`, `results_pins_arcface.json`, `results_insightface_bench_arcface.json`

App selects model via `APP_CKPT` env var (already supported in `application/backend/main.py:_config`). To use new model: `APP_CKPT=training_pipeline/checkpoints/best_arcface.pt uvicorn application.backend.main:app`.

**Enrollment compatibility:** Enabling TTA changes embedding values for the same input (it's the L2-normalized sum of a face and its flip, not the face alone). Vectors enrolled with old `embed_normalized` will not match `embed_tta` vectors. When switching to the ArcFace model with TTA on, the app must clear or re-enroll existing entries. Implementation: when the embedder loads, write the model SHA + tta-on flag to a sidecar `meta.json` next to `vectors.npy` (the actual store file in `application/backend/db.py:19`, NOT `embeddings.npy`). On startup, if the meta differs from the current model+TTA combo, refuse to load `vectors.npy` and instruct the user to re-enroll. This avoids silent score drift.

### ArcFace head — `training_pipeline/src/arcface_head.py` (~60 LOC, new file)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceHead(nn.Module):
    """
    Standard ArcFace head. Embeddings must arrive L2-normalized.
    Returns scaled logits ready for cross-entropy.
    """
    def __init__(self, embedding_dim: int, num_classes: int,
                 s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)  # wrong-hemisphere boundary
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, D] (already L2-normalized)
        # weight: [C, D] → L2-normalize columns for cosine similarity
        weight_norm = F.normalize(self.weight, dim=1)
        cos_theta = embeddings @ weight_norm.t()                 # [B, C]
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # cos(θ_y + m) = cos_θ·cos_m − sin_θ·sin_m
        sin_theta_y = torch.sqrt(1.0 - cos_theta.pow(2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta_y * self.sin_m

        # Standard ArcFace fallback (NOT easy-margin):
        # when cos_θ ≤ cos(π−m), use cos_θ − sin(π−m)·m instead of cos(θ+m).
        # This keeps the margin direction stable in the wrong-hemisphere region.
        mm = math.sin(math.pi - self.m) * self.m
        cos_theta_m = torch.where(
            cos_theta > self.threshold,
            cos_theta_m,
            cos_theta - mm,
        )

        # Inject margin only on the correct class
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        return logits * self.s

    @torch.no_grad()
    def center_norms(self) -> torch.Tensor:
        """For sanity logging — return per-class L2 norms of the weight matrix."""
        return self.weight.norm(dim=1)
```

Math is standard InsightFace ArcFace (non-easy-margin variant from the paper). The wrong-hemisphere fallback (`cos_θ − mm` when `cos_θ ≤ cos(π−m)`) handles negative cosines that arise early in training, before embeddings cluster — without it the gradient direction flips and loss can NaN.

### Training loop — `training_pipeline/src/train.py` changes

P1 stays unchanged (CE warmup, AdamW, classifier head, 3 epochs).

P2 changes:

```python
# Construct ArcFace head, get optimizer
arc_head = ArcFaceHead(
    embedding_dim=cfg["train"]["embedding_dim"],
    num_classes=num_ids,
    s=cfg["train"]["arcface_s"],
    m=cfg["train"]["arcface_m"],
).to(device)

# Switch to SGD for P2 (canonical ArcFace recipe)
p2_optim = torch.optim.SGD(
    [
        {"params": model.parameters(), "lr": cfg["train"]["p2_lr"]},
        {"params": arc_head.parameters(), "lr": cfg["train"]["p2_lr"]},
    ],
    momentum=0.9,
    weight_decay=cfg["train"]["weight_decay"],
    nesterov=False,
)

# Step decay schedule
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    p2_optim,
    milestones=cfg["train"]["p2_lr_decay_epochs"],
    gamma=cfg["train"]["p2_lr_decay_gamma"],
)

# Per-batch in P2:
emb = model.embed_normalized(images)           # L2-normalized [B, 512]
logits = arc_head(emb, labels)                  # ArcFace logits [B, C]
loss = F.cross_entropy(logits, labels)
```

Existing collapse gate continues to compute `spread = pos_sim_mean − neg_sim_mean` on a held-out batch each epoch end. If `spread < 0.05`, log warning and skip best.pt promotion for that epoch (same logic as current code).

NEW per-epoch logging:
```python
center_norms = arc_head.center_norms()
print(f"P2 e{epoch}: center_norm mean={center_norms.mean():.3f} "
      f"min={center_norms.min():.3f} max={center_norms.max():.3f}")
```

Sanity range for center norms is roughly [0.5, 2.0]. Outside that range signals head degeneracy (centers all collapsing to one direction, or weight blow-up).

**Checkpoint save points** (preserve existing two-file convention):

1. **Best, guarded** (`train.py:231-233`): only inside `if metrics["mean_acc"] > best_acc and metrics["spread"] > 0.05:`. This is the file the app loads. Format unchanged: `{"model": model.state_dict(), "cfg": cfg}`.
2. **Last, unguarded** (`train.py:235`): always written at end of training to `ckpt_dir / "last.pt"`. Same format. This file stays exactly as today — analysis-only, never promoted.
3. **ArcFace head** (new): once at end of training, write `arc_head.state_dict()` to `ckpt_dir / "arcface_head.pt"` (analysis only).

**Required train.py fix:** `train.py:233` hardcodes `"best.pt"` as the *guarded best-save* filename. Replace **only that line** with `ckpt_dir / cfg.get("checkpoint_name", "best.pt")`. Do **not** touch line 235 (`last.pt` keeps its literal name — it's a separate "last epoch" snapshot for analysis, not a competing best). With `checkpoint_name: best_arcface.pt` in the new YAML, the guarded best-save writes to `best_arcface.pt`; the always-at-end `last.pt` still gets written and never overwrites best.

### TTA — wired at every embedding choke point

**Method on the model** (`training_pipeline/src/model.py`):
```python
def embed_tta(self, x: torch.Tensor) -> torch.Tensor:
    """Average of normalized embeddings from x and its horizontal flip.

    Cost: 2× forward pass at inference. Helps on profile/pose benchmarks
    by reducing left/right asymmetry. Free improvement, no retraining.
    """
    e1 = self.embed_normalized(x)
    e2 = self.embed_normalized(torch.flip(x, dims=[-1]))
    return F.normalize(e1 + e2, dim=-1)
```

**Shared evaluator** (`training_pipeline/src/eval_lfw.py:evaluate_lfw`) currently hardcodes `model.embed_normalized(x)` at line 164. Add `use_tta: bool = False` parameter; inside the loop, branch:
```python
embed_fn = model.embed_tta if use_tta else model.embed_normalized
embs_list.append(embed_fn(x).cpu())
```

**Eval CLIs** (`evaluation/eval_lfw.py`, `eval_pins.py`, new `evaluation/benchmark_insightface.py`) gain `--tta` (default False for back-compat with existing baselines). The flag is forwarded to `evaluate_lfw(..., use_tta=args.tta)` and the Pins/benchmark equivalents.

**App embedder** (`application/backend/inference.py:Embedder`) gains a `use_tta: bool = False` constructor arg, stored as `self.use_tta`. `embed()` switches:
```python
embed_fn = self.model.embed_tta if self.use_tta else self.model.embed_normalized
return embed_fn(face_tensor).squeeze(0).cpu()
```

App wires TTA from a new env var. Add to `_config()` in `application/backend/main.py`:
```python
"use_tta": os.environ.get("APP_TTA", "0") == "1",
```
And pass it: `Embedder(cfg["checkpoint"], device="cpu", use_tta=cfg["use_tta"])`. Default off (back-compat); deployment with ArcFace sets `APP_TTA=1`.

### Config — `training_pipeline/configs/train_arcface.yaml` (new file)

```yaml
manifest: process-data/manifest.parquet
checkpoints_dir: training_pipeline/checkpoints
tensorboard_dir: training_pipeline/tensorboard_logs
seed: 42
checkpoint_name: best_arcface.pt

train:
  phase1_epochs: 3
  phase2_epochs: 27
  batches_per_epoch: 1500
  phase1_batch_size: 128
  p: 32
  k: 4
  # P1 (CE warmup): AdamW
  lr_head: 5.0e-4
  lr_backbone: 1.0e-4
  weight_decay: 5.0e-4
  warmup_steps: 500
  num_workers: 8
  embedding_dim: 512
  # P2 (ArcFace): SGD
  p2_lr: 0.01
  # Milestones are P2-relative (scheduler.step() called once per P2 epoch).
  # Canonical ArcFace decay is at 62.5% and 87.5% of total run; mapped to P2:
  # 30 total * 0.625 - 3 P1 = 16.75 → 17; 30 total * 0.875 - 3 P1 = 23.25 → 24.
  p2_lr_decay_epochs: [17, 24]
  p2_lr_decay_gamma: 0.1
  arcface_s: 64.0
  arcface_m: 0.5

eval:
  pairs_txt: preprocess-data/lfw/pairs.txt
  max_pairs_inloop: null
  batch_size: 128
```

Trainer accepts `--config` CLI arg (already supported). Default stays `train.yaml`. ArcFace runs use `--config training_pipeline/configs/train_arcface.yaml`.

### Smoke test — `training_pipeline/tests/test_arcface_smoketest.py` (new file)

Lives under `training_pipeline/tests/` because `pyproject.toml` testpaths is `["training_pipeline/tests", "application/tests"]` — a top-level `tests/` directory would not be collected by pytest.

Mirrors existing `training_pipeline/tests/test_two_phase_smoketest.py`. Constructs:
- 8-identity × 8-image synthetic dataset (random tensors with per-class bias for separability)
- Reduced config: phase1_epochs=2, phase2_epochs=3, batches_per_epoch=10
- Same trainer entry point with new `train_arcface.yaml`-style overrides

Assertions:
- All losses are finite (no NaN/inf) across every P1+P2 batch.
- ArcFace logits are finite. Upper bound is `s` (cosine ≤ 1). Lower bound is `−(1 + mm) · s` where `mm = sin(π − m) · m ≈ 0.240` for m=0.5 — fallback branch can push `cos_θ − mm` to `−1.240`, scaled to `−s·1.240 ≈ −79.4`. Assert `logits.min() >= -(1.0 + mm) * s - 1e-3` (small slack for fp16/fp32 noise). Do **not** assert tight `[-s, s]`; that bound is wrong for the standard non-easy fallback.
- Center norms stay in `[0.3, 3.0]` for every P2 epoch (wide sanity window, tight enough to catch blow-ups).
- Final embedding spread on a held-out batch > 0.05 (matches the production collapse-gate threshold).
- `best_arcface.pt` is written under the checkpoints dir (file exists, loadable via `torch.load`, has both `model` and `cfg` keys).

Runs in <2 minutes on CPU. Gates the change in CI.

## Eval protocol changes

`evaluation/eval_lfw.py` and `eval_pins.py` exist; `evaluation/benchmark_insightface.py` does **not** yet exist in the repo (the 8-suite numbers in `results_insightface_bench.json` were generated by a one-off VM script that was never committed). This spec requires creating it.

After training, run three evaluators against `best_arcface.pt`:

1. **Strict LFW 10-fold** (strict mode is already always on; flag-less). Use the real `--checkpoint`/`--out` CLI:
   ```
   python -m evaluation.eval_lfw \
       --checkpoint training_pipeline/checkpoints/best_arcface.pt \
       --tta \
       --out evaluation/results_arcface.json
   ```
2. **Pins cross-dataset** — must point `--lfw-results` to the new file so the threshold comes from the ArcFace model, not the old triplet baseline:
   ```
   python -m evaluation.eval_pins \
       --checkpoint training_pipeline/checkpoints/best_arcface.pt \
       --tta \
       --lfw-results evaluation/results_arcface.json \
       --out evaluation/results_pins_arcface.json
   ```
3. **8-suite InsightFace benchmarks** — script is new (see "File changeset"):
   ```
   python -m evaluation.benchmark_insightface \
       --checkpoint training_pipeline/checkpoints/best_arcface.pt \
       --tta \
       --out evaluation/results_insightface_bench_arcface.json
   ```

Existing scripts gain `--tta` (flag, default off — preserves the meaning of the existing `results.json` numbers). `eval_lfw.py` does not need a `--strict` flag because it already passes `strict=True` unconditionally to `evaluate_lfw` (see `evaluation/eval_lfw.py:46`).

Notebook `evaluation/benchmarks.ipynb` adds a new section "Triplet vs ArcFace" that loads both `results_insightface_bench.json` and `results_insightface_bench_arcface.json` and renders a side-by-side comparison: bars per benchmark with two colors per bar group, and a Δ-vs-triplet column.

## Success criteria

`best_arcface.pt` is promoted to canonical (replaces `application/models/best.pt`) only if all four are true:

1. **LFW strict 10-fold acc > 95%** — must not regress vs current 95.03%
2. **CFP-FP acc > 88%** — must not regress vs current 88.27%
3. **AgeDB-30 acc > 80%** — target improvement vs current 76.03%
4. **Spread on LFW > 0.7** — embedding quality holds vs current 0.72

If 1–2 regress: abort. Don't promote. Keep `best.pt` as canonical. Document in `docs/cross-dataset-eval.md`.

If 1–2 hold but 3 does not improve: don't promote, but keep both .pt files for A/B comparison. Document the failure mode (likely training data or scale).

If all four hold: promote `best_arcface.pt` to `application/models/best.pt`, update `docs/cross-dataset-eval.md` with new numbers, regenerate notebook figures.

## Risks + mitigations

1. **ArcFace late-epoch collapse.** Literature warns "70% LFW, 5% validation" failure mode when margins too high / LR not decayed. *Mitigation:* CE warmup (3 epochs) + SGD step decay at epochs 20, 27 + collapse gate + center-norm logging.
2. **AdamW→SGD optimizer switch can hit different local minima.** *Mitigation:* SGD only in P2; P1 stays AdamW so warmup remains stable.
3. **TTA cost at inference.** 2× forward pass. *Mitigation:* still <50ms on M4 for a single face; acceptable for app latency.
4. **Wrong-hemisphere fallback interaction with cosine clamp.** Standard `cos(θ+m)` formula is only applied when `cos_θ > cos(π−m) ≈ −0.878`. Otherwise the head returns `cos_θ − mm` (where `mm = sin(π−m)·m ≈ 0.240`), keeping the gradient on the same side of the boundary. *Mitigation:* xavier_normal_ init keeps centers well-spread initially; CE warmup ensures embeddings are already clustered into the correct hemisphere before P2 starts.

## File changeset

**New files:**
- `training_pipeline/src/arcface_head.py`
- `training_pipeline/configs/train_arcface.yaml`
- `training_pipeline/tests/test_arcface_smoketest.py` (under existing testpath)
- `evaluation/benchmark_insightface.py` (does not exist in repo today; produced the 8-suite JSON only on the VM)

**Modified files:**
- `training_pipeline/src/train.py` — (a) P2 swap to ArcFace head + SGD m=0.9 + `MultiStepLR` step decay + per-epoch center-norm logging; (b) replace hardcoded `"best.pt"` at line 233 with `cfg.get("checkpoint_name", "best.pt")` so the new YAML drives the save path
- `training_pipeline/src/model.py` — add `embed_tta` method
- `training_pipeline/src/eval_lfw.py` — add `use_tta` param to `evaluate_lfw`, branch the embedding call at line 164
- `evaluation/eval_lfw.py` — add `--tta` CLI flag, forward as `use_tta=args.tta`
- `evaluation/eval_pins.py` — add `--tta` CLI flag, forward through the pins evaluator
- `application/backend/inference.py` — add `use_tta` ctor arg to `Embedder`, branch in `embed()`
- `application/backend/main.py` — read `APP_TTA` env var in `_config()`, pass `use_tta` to `Embedder`; add embedding-store version guard (model SHA + tta flag) that refuses to load a mismatched enrollment DB
- `evaluation/benchmarks.ipynb` — add "Triplet vs ArcFace" comparison section
- `application/Dockerfile` — fix stale `ENV APP_THRESHOLD=0.5` (line 16) to `0.565` to match the new backend/UI default; without this, containerized runs silently regress to the old threshold and skew any ArcFace deployment evaluation
- `application/frontend/app.js` — guard `snapBase64()` against zero-dimension video (camera denied or `loadedmetadata` not fired): if `video.videoWidth === 0 || video.videoHeight === 0`, throw a clear error (`"Camera not ready — grant permission or wait for stream"`) and surface it in the result UI instead of POSTing an empty frame to `/enroll` or `/verify`

**Result files (generated, gitignored or committed depending on size):**
- `evaluation/results_arcface.json`
- `evaluation/results_pins_arcface.json`
- `evaluation/results_insightface_bench_arcface.json`
- `training_pipeline/checkpoints/best_arcface.pt` (94MB, gitignored, kept on VM + local)
- `training_pipeline/checkpoints/arcface_head.pt` (small, kept on VM)

## Out of scope

- VGGFace2 data addition (separate spec if needed)
- Sub-center ArcFace
- Adaptive margin scheduling
- Adversarial training (talfw mitigation)
- New model backbones
- Mobile deployment / quantization
