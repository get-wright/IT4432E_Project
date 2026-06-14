# AdaFace Model — Architecture, Training, and Results

This document covers the **IResNet50 + AdaFace** model that is deployed in the app (`application/models/best.pt`). This is the model actually trained for the project — see `training-overview.md` for the earlier two-phase CE+triplet baseline that it replaced.

**Download the pre-trained checkpoint (≈167 MB):** https://drive.google.com/file/d/1SMPbpQ60rQEJeNORMB5q4jn_7u6zU8-I/view?usp=sharing — place it at `application/models/best.pt` (or run `gdown 1SMPbpQ60rQEJeNORMB5q4jn_7u6zU8-I -O application/models/best.pt`).

---

## Architecture

**Backbone: IResNet50**

The InsightFace ResNet-50 variant. Key differences from torchvision's standard ResNet50:

| Feature | torchvision ResNet50 | IResNet50 |
|---|---|---|
| Residual block | Standard (BN→Conv→BN→ReLU→Conv→BN) | Pre-activation (`IBasicBlock`: BN→Conv→BN→PReLU→Conv→BN) |
| Activation | ReLU | PReLU (learnable slope, per-channel) |
| Input stride | 7×7 conv, stride 2 | 3×3 conv, stride 1 — preserves spatial resolution at layer 1 |
| Global pool | AdaptiveAvgPool2d → flatten | No pool — flatten 7×7×512 → 25088 directly |
| Embedding head | `Linear(2048, emb_dim)` | `Linear(25088, 512)` → `BatchNorm1d(512)` |
| Layer config | [3, 4, 6, 3] | [3, 4, 14, 3] — deeper layer 3 |

The `BatchNorm1d` at the end is important: its output is the raw embedding passed to the AdaFace loss during training; at inference the output is L2-normalised to the unit hypersphere before any cosine comparison.

**Embedding: 512-d, L2-normalised at inference**

```
input (3, 112, 112)
  → IResNet50 backbone
  → Linear(25088, 512) + BatchNorm1d(512)
  → L2 normalise  ← only at inference / in the loss, not inside forward()
  → 512-d unit vector
```

Total parameters: ~43.6M (trainable: ~43.5M — BN weight in embedding head is frozen to 1.0 by design).

---

## Loss: AdaFace

**Reference:** Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition", CVPR 2022.

Standard ArcFace uses a fixed angular margin `m` for every sample. AdaFace makes the margin *adaptive*: samples with a high feature norm (confident, sharp, well-lit faces) receive a *larger* penalty; samples with a low norm (blurry, occluded, dark) receive a *smaller* penalty.

The mechanism:

1. The BN embedding head outputs a vector whose **L2 norm** correlates with image quality (brighter, sharper images → higher activation → higher norm).
2. The loss computes a per-batch running mean and std of these norms.
3. A `margin_scaler` is derived from how each sample's norm compares to the batch mean.
4. The angular margin `g_angular` and additive offset `g_add` are scaled by `margin_scaler`.

Net effect: **the model implicitly mines hard, high-quality examples** without an explicit hard-mining step. This is why SLLFW (lookalikes) and AgeDB (hard cross-age pairs) score significantly better than a plain triplet or fixed-ArcFace baseline.

**Hyperparameters used:**

| Parameter | Value | Role |
|---|---|---|
| `m` | 0.4 | Base angular margin magnitude |
| `h` | 0.333 | Scaling factor for the adaptive component |
| `s` | 64.0 | Logit scale (temperature) |
| `t_alpha` | 0.01 | EMA rate for running batch norm/mean |

---

## Training

### Data

**CASIA-WebFace** — 10,572 identities, ≈490K images. Shipped as InsightFace MXNet RecordIO (`.rec`/`.idx`/`property`). Read with a pure-Python RecordIO parser in `train_local.py` — no MXNet dependency.

Input preprocessing:
- Decode JPEG → BGR → RGB uint8
- `RandomHorizontalFlip`
- `ToTensor` → normalise to `[-1, 1]` (mean=std=0.5)

Aligned to 112×112 during dataset construction (the RecordIO images are already cropped/aligned in the InsightFace format).

### Recipe

```
Epoch 1–30, batch=256, SGD lr=0.05, momentum=0.9, wd=5e-4
  LR warmup: linear ramp over epoch 1
  LR decay:  MultiStep ×0.1 at epochs 16, 24, 28
  Grad clip: max_norm=5.0
  Mixed precision: FP16 (torch.amp.autocast + GradScaler)
  Hardware:  NVIDIA RTX 4070 12 GB
```

For each batch step:
1. Forward: `emb = model(imgs)` → raw embedding
2. Compute norms: `norms = torch.norm(emb, 2, dim=1, keepdim=True)`
3. AdaFace head: `logits = head(emb / norms, norms, labels)` — head normalises kernel too
4. Loss: `F.cross_entropy(logits, labels)`
5. `scaler.scale(loss).backward()` → unscale → clip → step

Every 2 epochs: LFW 10-fold verification probe on the raw LFW deepfunneled images (no separate alignment pipeline required at eval time).

End of training: save `ckpt/embedder.pt` — model weights only, no head, no optimiser.

### Training curve

| Epoch | Train loss | LFW acc (approx, raw) |
|---|---|---|
| 1  | ~3.2 | ~78% |
| 8  | ~1.9 | ~84% |
| 16 | ~1.3 | ~90% |
| 24 | ~0.8 | ~94% |
| 28 | ~0.5 | ~97% |
| 30 | ~0.4 | ~99% |

LFW is evaluated using raw deepfunneled images during training (no MTCNN alignment). The formal 10-fold eval in `evaluation/results_adaface.json` uses pre-aligned InsightFace bins and reports higher numbers.

---

## Evaluation results

Full protocol: InsightFace pre-aligned 112×112 `.bin` files, 10-fold CV threshold per benchmark. All metrics in `evaluation/results_adaface.json`. Charts in `evaluation/adaface_evaluation.ipynb`.

### 8-benchmark summary

| Benchmark | What it tests | Acc (CV) | Std | F1 | ROC-AUC | Spread |
|---|---|---|---|---|---|---|
| **lfw** | Standard frontal verification | **99.30%** | ±0.40% | 0.9938 | 0.9995 | 0.582 |
| **cfp_ff** | Frontal vs frontal | **99.47%** | ±0.26% | 0.9949 | 0.9996 | 0.615 |
| **cfp_fp** | Frontal vs profile (±90°) | **95.03%** | ±1.08% | 0.9491 | 0.9766 | 0.395 |
| **agedb_30** | Same person, 30-year age gap | **94.25%** | ±1.26% | 0.9443 | 0.9831 | 0.345 |
| **calfw** | Cross-age LFW | **93.48%** | ±0.97% | 0.9334 | 0.9732 | 0.415 |
| **cplfw** | Cross-pose LFW (extreme yaw) | **89.28%** | ±1.60% | 0.8879 | 0.9388 | 0.319 |
| **sllfw** | Similar-looking lookalikes | **98.05%** | ±0.60% | 0.9815 | 0.9962 | 0.478 |
| talfw | Transfer-attack (adversarial) | 50.00% | ±0.00% | 0.667 | 0.417 | −0.073 |

**Spread** = `pos_sim_mean − neg_sim_mean`. A higher spread means the embedding space is more discriminative, independent of the chosen threshold.

### LFW confusion matrix (τ = 0.237)

```
                Predicted: Different   Predicted: Same
Actual: Different        2974                  26
Actual: Same               26                2974
```

TP=2974  FP=26  TN=2974  FN=26 (out of 6000 pairs, 3000 pos + 3000 neg)

### Why the numbers are better than the baseline

The old CE+triplet model (ResNet50, `training-overview.md`) achieved 90.90% LFW. Key differences that explain the ~8.4 pp gap:

| Factor | Old model | This model |
|---|---|---|
| Architecture | torchvision ResNet50 + `Linear(2048,512)` | IResNet50 (pre-activation, deeper, 25088-d flat) |
| Loss | Fixed soft-margin triplet | AdaFace adaptive margin — hard examples weighted by norm |
| Input size | 160×160, ImageNet normalisation | 112×112, [-1,1] normalisation (standard for face models) |
| Epochs | 20 (3 warmup + 17 triplet) | 30 |
| Batch | 128 (PKSampler, P=32, K=4) | 256 (standard shuffle) |
| LR | AdamW, 1e-4/5e-4, cosine | SGD, 0.05, MultiStep |

The dominant factor is the loss: AdaFace's adaptive margin provides hard-example weighting that the soft-margin triplet loss cannot match.

---

## App integration

The checkpoint is deployed via `application/backend/inference.py`. The `Embedder` class auto-detects the format:

```python
ckpt = torch.load(checkpoint, ...)
if "cfg" in ckpt:
    # Old IT4432E training pipeline — ResNet50 + FaceEmbedding
    model = FaceEmbedding(...)
else:
    # train_local.py AdaFace — IResNet50
    model = iresnet50(embedding_size=ckpt["model"]["fc.weight"].shape[0])
```

`FaceAligner` (`face_align.py`) outputs 112×112 tensors normalised to `[-1, 1]` (mean=std=0.5), matching AdaFace training. MTCNN is used for face detection and alignment.

**Deployment threshold:** The app default is `APP_THRESHOLD=0.5`. For stricter verification set it to `0.7+`; for more permissive enroll/demo use `0.3–0.4`. The LFW-optimal threshold is `τ = 0.237` but that's tuned on a benchmark with clean aligned images — real-world webcam images warrant a lower threshold.

---

## Limitations

| Failure mode | Why | Mitigation |
|---|---|---|
| Adversarial inputs (TALFW: 50%) | No adversarial training; small perturbations invert cosine ordering | Adversarial training (PGD), input preprocessing (bit-depth reduction) |
| Large age gaps (AgeDB: 94%, still good) | CASIA-WebFace lacks same-identity images across decades | Add MS-Celeb-1M or age-progression dataset |
| Extreme yaw (CPLFW: 89%) | CASIA is mostly frontal | 3D-augmented training data or multi-pose aggregation at inference |
| Lookalikes (SLLFW: 98%, excellent) | Handled well by AdaFace adaptive margin | — |
