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
- AMP + `torch.cuda.amp.GradScaler` (facenet-pytorch torch 2.2 pin)
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

App selects model via `APP_MODEL_PATH` env var (default `application/models/best.pt`). To use new model: `APP_MODEL_PATH=training_pipeline/checkpoints/best_arcface.pt`.

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
        self.threshold = math.cos(math.pi - m)  # easy-margin guard
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

        # Easy-margin guard: only apply margin where cos_θ > threshold,
        # else fall back to cos_θ (prevents margin pushing into wrong hemisphere).
        cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta)

        # Inject margin only on the correct class
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        return logits * self.s

    @torch.no_grad()
    def center_norms(self) -> torch.Tensor:
        """For sanity logging — return per-class L2 norms of the weight matrix."""
        return self.weight.norm(dim=1)
```

Math is standard InsightFace ArcFace. The `easy-margin` guard handles negative cosines (which can arise early in training before embeddings cluster) — without it the loss can NaN.

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

At end of training: save `model.state_dict()` to `best_arcface.pt`, save `arc_head.state_dict()` to `arcface_head.pt`. The app loads only the model state dict.

### TTA — `training_pipeline/src/model.py` adds method

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

Eval scripts gain a `--tta` CLI flag (default False for back-compat). When set, embed function used is `embed_tta` instead of `embed_normalized`. Affects all 3 evaluators (`eval_lfw.py`, `eval_pins.py`, `benchmark_insightface.py`).

App backend (`application/backend/main.py`) calls `embed_tta` unconditionally — TTA is on by default in deployment.

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

### Smoke test — `tests/test_arcface_smoketest.py` (new file)

Mirrors `tests/test_two_phase_smoketest.py`. Constructs:
- 8-identity × 8-image synthetic dataset (random tensors with per-class bias for separability)
- Reduced config: phase1_epochs=2, phase2_epochs=3, batches_per_epoch=10
- Same trainer entry point

Assertions:
- No NaN losses anywhere
- ArcFace logits `.max()` ≈ scale (within ±10% of `s=64`)
- Center norms stay in [0.5, 2.0] for every epoch
- Final embedding spread on a held-out batch > 0.1
- `best_arcface.pt` exists after training

Runs in <2 minutes on CPU. Gates the change in CI.

## Eval protocol changes

After training, run three evaluators against `best_arcface.pt`:

1. **Strict LFW 10-fold:** `python -m evaluation.eval_lfw --ckpt training_pipeline/checkpoints/best_arcface.pt --strict --tta --out evaluation/results_arcface.json`
2. **Pins cross-dataset:** `python -m evaluation.eval_pins --ckpt ... --tta --out evaluation/results_pins_arcface.json`
3. **8-suite InsightFace benchmarks:** `python -m evaluation.benchmark_insightface --ckpt ... --tta --out evaluation/results_insightface_bench_arcface.json`

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
4. **Easy-margin guard interaction with cosine clamp.** Margin only applied when `cos_θ > cos(π-m) ≈ -0.878`. If embeddings cluster wrongly early (cos_θ < -0.878), the head behaves as pure cosine, gradient still flows. *Mitigation:* xavier_normal_ init keeps centers well-spread initially.

## File changeset

**New files:**
- `training_pipeline/src/arcface_head.py`
- `training_pipeline/configs/train_arcface.yaml`
- `tests/test_arcface_smoketest.py`

**Modified files:**
- `training_pipeline/src/train.py` — P2 swap to ArcFace + SGD + step decay + center logging
- `training_pipeline/src/model.py` — add `embed_tta` method
- `evaluation/eval_lfw.py` — add `--tta` flag
- `evaluation/eval_pins.py` — add `--tta` flag
- `evaluation/benchmark_insightface.py` — add `--tta` flag
- `application/backend/main.py` — use `embed_tta` instead of `embed_normalized`
- `evaluation/benchmarks.ipynb` — add "Triplet vs ArcFace" comparison section

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
