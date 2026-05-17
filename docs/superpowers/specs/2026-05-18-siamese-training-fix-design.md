# Siamese Training Fix — Design

## Background

The first training run (commit `b62c6c1`..`32b4ce5`, branch `feat/face-recognition-siamese`) collapsed silently:

- Every face crop maps to nearly the same 512-d vector at inference. Cosine similarity is ≈ 0.997 between any two faces, regardless of identity.
- The reported "97.48% LFW accuracy" was a label-leak artifact. `load_pairs_txt` silently dropped pairs whose MTCNN-aligned image was missing; the surviving 2378 pairs were 2318 positive + 60 negative (97.48% positives). Threshold sweep picked `t = -1.0` (predict "same" for everything) → accuracy = positive ratio.
- The in-loop LFW probe during training had the same bug. Training had no real feedback signal.
- Cross-dataset test on Pins (105 disjoint celebrities) gave 50.1% — i.e. random.

Triplet loss collapsed in a textbook way:

1. Batch-hard mining from a near-random ImageNet-pretrained backbone always returns nonsense as "hardest negative" — the model can't find a useful gradient, settles on a degenerate constant-vector solution.
2. With L2-normalize at the output, max possible `d_an - d_ap` is 2 and margin is 0.3 — collapse is the easiest minimum.
3. Backbone learning rate (3e-5) was effectively frozen; the head alone couldn't escape collapse.
4. `RandomErasing(p=0.2)` on 160×160 aligned crops erased eyes/nose/mouth at random, removing the identity signal the model was supposed to learn.

We keep the Siamese-twin-network + triplet-loss paradigm (it's the course assignment). The fix is to apply the canonical FaceNet/MassFace recipe: softmax warmup → semi-hard triplet, soft-margin, L2-norm at eval only, unfrozen backbone, drop harmful augmentation, repair the eval.

## Goals

- Train a Siamese face embedding network that **actually discriminates** — pos/neg cosine similarity spread > 0.3 on LFW.
- LFW 10-fold verification accuracy > 80% on the full 6000 pairs, with a non-trivial threshold (`0.0 < t < 0.9`).
- Cross-dataset accuracy on Pins > 70% with the same threshold — confirms generalization, not LFW-specific overfit.
- Eval pipeline that **cannot silently leak** — every collapse-pattern checked with assertions before any results.json is written.

## Non-goals

- Switching loss to ArcFace, CosFace, or any non-pair-based variant.
- Using a pretrained face embedder (`InceptionResnetV1(pretrained='vggface2')` etc).
- Adding additional datasets (CelebA stays dropped; VGGFace2 / MS1M not in scope).
- Changing the inference path of the FastAPI app — same `/enroll`, `/verify` contracts.

## Architecture overview

```
Phase 1 (epochs 1–3, softmax warmup):

  image → backbone (ResNet50, unfrozen) → embedding head (Linear 2048→512)
                                                ↓ (no L2 norm)
                                          classifier head (Linear 512→1249)
                                                ↓
                                          CrossEntropy(logits, identity_label)

Phase 2 (epochs 4–20, semi-hard triplet):

  image → backbone → embedding head → raw 512-d vector
                                          ↓
                                   F.normalize inside loss  ← matches FaceNet; prevents norm-scale escape
                                          ↓
                      semi-hard triplet selection within PK batch (Euclidean on unit-sphere)
                                          ↓
                                  softplus(d_ap - d_an)

Inference / eval:

  image → backbone → embedding head → F.normalize(out, p=2, dim=1) → unit-norm embedding
  cosine_sim(a, b) = a · b
```

**Why L2-normalize inside the loss (not at the model output):** Training on un-normalized embeddings lets the model satisfy `softplus(d_ap - d_an)` by scaling overall embedding norms, not by improving angular identity structure. At eval we normalize and that fake "progress" disappears. FaceNet (Schroff 2015) computes its triplet loss on L2-normalized embeddings for exactly this reason. We keep the model's `forward` un-normalized so `embed_normalized()` stays explicit and the Phase 1 classifier head sees raw vectors (better with CE); the Phase 2 triplet loss explicitly L2-normalizes its input on entry.

## Components

### `training_pipeline/src/model.py`

- `FaceEmbedding.forward(x)` returns **un-normalized** 512-d vector. (Previously normalized.)
- New `FaceEmbedding.embed_normalized(x)`: forward + L2-normalize. Used by eval and by app inference. The triplet loss in Phase 2 also internally L2-normalizes its input (Euclidean distance on **unit-normalized** vectors). Phase 1's classifier head sees raw forward output — CE benefits from un-normalized logits.
- New module `ClassifierHead(nn.Module)`:
  - Single `nn.Linear(embedding_dim, n_identities)`, no bias.
  - Used only during Phase 1; instantiated separately from `FaceEmbedding` so it can be cleanly dropped at handoff.

### `training_pipeline/src/loss.py`

- Keep `_pairwise_dist` (correct, unchanged).
- New `semi_hard_triplet_loss(emb, labels, margin=0.3) -> tuple[Tensor, int]`:
  - **First line: `emb = F.normalize(emb, p=2, dim=1)`** — distances computed on the unit hypersphere, no norm-scale escape.
  - For each (anchor a, positive p) where `labels[a] == labels[p]` and `a != p`:
    1. Compute `d_ap = dist[a, p]`.
    2. Build mask over j ≠ a where `labels[j] != labels[a]` and `d_ap < dist[a, j] < d_ap + margin`.
    3. If mask non-empty → sample one negative uniformly.
    4. Elif any j with `dist[a, j] > d_ap` → take argmin of those (closest harder-than-positive negative).
    5. Else → skip this anchor.
  - Collected triplets pass through `softplus_loss(d_ap, d_an) = log(1 + exp(d_ap - d_an))`. Mean over selected triplets.
  - Returns `(loss, n_triplets)` so the train loop can skip `optimizer.step()` cleanly when `n_triplets == 0`.
  - **No-triplets contract:** if no anchor finds a valid negative, returns `(emb.sum() * 0.0, 0)` — a properly-graph-connected zero so `loss.backward()` is safe but produces no gradients. Caller checks `n_triplets > 0` before stepping.
- New `softplus_loss(d_ap, d_an)` — small helper, one-line `F.softplus(d_ap - d_an)`.
- Keep `batch_hard_triplet_loss` available for ablation but unused in the main run.

### `training_pipeline/src/dataset.py`

- `train_transform()`: **drop `RandomErasing`.** Keep `Resize(160,160) + HorizontalFlip + ColorJitter(0.2) + ToTensor + Normalize(ImageNet)`.
- `eval_transform()` unchanged.
- `FaceDataset` unchanged.
- `PKSampler` unchanged (still needed for Phase 2; Phase 1 uses the default `RandomSampler`).

### `training_pipeline/src/train.py`

Rewrite into two sequential phase loops driven by config:

- `cfg.train.phase1_epochs = 3`, `cfg.train.phase2_epochs = 17`, `cfg.train.batches_per_epoch = 1500`.
- `cfg.train.phase1_batch_size = 128` (used by Phase 1 RandomSampler with replacement to force exact `batches_per_epoch` length per epoch).
- `cfg.train.p = 32`, `cfg.train.k = 4` (PKSampler for Phase 2; batch size = P×K = 128).
- `cfg.train.lr_backbone = 1.0e-4`, `cfg.train.lr_head = 5.0e-4`, `cfg.train.weight_decay = 5.0e-4`.
- `cfg.train.margin = 0.3` (used by semi-hard selection band, not by the loss hinge).

Phase 1 loop:
1. Build `FaceEmbedding` + `ClassifierHead(512, n_identities)` (n derived from training manifest).
2. DataLoader: `RandomSampler(dataset, replacement=True, num_samples=cfg.train.batches_per_epoch * cfg.train.phase1_batch_size)`, `batch_size=cfg.train.phase1_batch_size` (default 128). This makes "epoch" exactly `batches_per_epoch` batches regardless of dataset size, so LR schedule + warmup + phase-1 gate are deterministic.
3. AdamW over both `FaceEmbedding` and `ClassifierHead` with split learning rates.
4. Per batch: forward → classifier logits → CE → backward → step.
5. End of epoch: run LFW probe (full 6000 pairs, fallback to raw on MTCNN miss) **with `strict=False`** — no assertions, just metrics for logging. Append `{spread, accuracy, pos_mean, neg_mean}` to `history.json`.
6. **After all phase-1 epochs are done**, evaluate the spread once. If `spread > 0.05` → handoff to phase 2. If not → abort with `RuntimeError("phase 1 produced collapsed embeddings — fix data/env before continuing")`. Early epochs are allowed to be weak; the gate fires only once at the boundary.
7. Save `checkpoints/phase1_end.pt`.

Phase 2 loop:
1. Drop classifier head (delete reference, free parameters from optimizer).
2. Re-init optimizer with backbone + embedding head only, same split LRs.
3. Switch sampler to PKSampler (P=32, K=4) with `batches_per_epoch=1500`.
4. Cosine LR schedule across phase-2 epochs only, with 500-step warmup at the start of phase 2.
5. Per batch: forward → `loss, n_triplets = semi_hard_triplet_loss(emb, labels)` → `if n_triplets > 0: loss.backward(); optimizer.step()`. Mixed precision. Log running `n_triplets` per epoch — a chronic zero is a sampler/mining bug.
6. End of epoch: LFW probe **with `strict=False`** + history.json append. Update `checkpoints/best.pt` if this epoch's probe LFW > best so far AND `spread > 0.05` (don't promote a collapsed checkpoint to best, even if numerics happen to favor it).

After phase 2: write `checkpoints/last.pt`.

### `training_pipeline/src/eval_lfw.py`

**Pair object.** `load_pairs_txt` now returns `list[LfwPair]` where:

```python
@dataclass
class LfwPair:
    name1: str
    idx1: int
    name2: str
    idx2: int
    same: int  # 0 or 1
```

The evaluator constructs aligned + raw paths from these fields plus two explicit roots (`aligned_root`, `raw_root`) passed by the caller. No filtering on disk presence at load time — all 6000 pairs are returned.

**Aligned and raw roots:** discovered, not hardcoded.

- `aligned_root` defaults to `ROOT / "process-data/lfw_pairs"` (populated by `process.py`) but is exposed as a CLI flag `--aligned-root` and as a parameter to `evaluate_lfw`.
- `raw_root` defaults are discovered by reusing the existing helper logic in `process-data/process.py::build_pairs_lfw`, which already walks the Kaggle dump to find the identity-folder level — it handles `lfw-deepfunneled/`, `lfw_funneled/`, and double-nested layouts. Refactor that discovery into `process-data/lfw_layout.py::find_lfw_identity_root(raw_dir)`, import it in both `process.py` (preserving existing behaviour) and `eval_lfw.py`. CLI flag `--raw-root` overrides.

For each `LfwPair`, path construction is `<root>/<name>/<name>_<idx:04d>.jpg` for both aligned and raw roots. `_load_image_with_fallback(aligned, raw)`:
- If `aligned.exists()` → load aligned, apply `eval_transform`.
- Else → load raw, center-crop to `0.8 * min(w, h)` square, resize 160×160, apply `eval_transform`.
- Asserts `raw.exists()` — if even raw is missing, that's a setup bug, not a runtime path. Raise.

**Sanity assertions split into two layers:**

```python
def _assert_distribution_sane(metrics):
    assert metrics['spread'] > 0.05, f"COLLAPSED: pos={metrics['pos_sim_mean']:.3f}, neg={metrics['neg_sim_mean']:.3f}"
    assert metrics['pos_sim_std'] > 0.01, f"COLLAPSED: pos std={metrics['pos_sim_std']:.4f} too tight"
    assert 0.4 < metrics['pos_ratio'] < 0.6, f"LABEL LEAK: {metrics['pos_ratio']:.2%} positive"


def _assert_threshold_sane(metrics):
    # Only applies when a threshold was selected via sweep (i.e. LFW 10-fold CV).
    # Cosine threshold should be solidly positive for a working face model. Range
    # matches the project goal: 0.0 < t < 0.9.
    assert 0.0 < metrics['threshold_global'] < 0.9, f"THRESHOLD AT BOUND: {metrics['threshold_global']}"


def evaluate_lfw(model, pairs, ..., strict: bool = True) -> dict:
    ...
    metrics = {...}  # always computed
    if strict:
        _assert_distribution_sane(metrics)
        _assert_threshold_sane(metrics)
    return metrics
```

- Training loop calls with `strict=False` — returns metrics dict including `spread`. Training itself decides whether to abort.
- Final CLI run (`python -m evaluation.eval_lfw`) calls with `strict=True`. Any failure → `RuntimeError`, no `results.json` written.
- `eval_pins.py` reuses only `_assert_distribution_sane` (no threshold sweep, no threshold check). See below.

### `evaluation/eval_lfw.py` (CLI)

Unchanged in structure; calls into `evaluate_lfw` which now enforces sanity. Output `results.json` adds `pos_sim_mean`, `pos_sim_std`, `neg_sim_mean`, `neg_sim_std`, `spread`, `n_pairs_total`, `n_pairs_used`, `n_aligned`, `n_raw_fallback`.

### `evaluation/eval_pins.py` (new)

Loads the already-downloaded Pins dataset, builds 1500 positive + 1500 negative pairs at fixed seed. Aligns each unique image once with MTCNN; falls back to center-crop raw on failure (same pattern as LFW eval). Embeds, computes cosine sim per pair.

**Crucially: uses the LFW-tuned threshold as a fixed parameter — does NOT tune its own threshold.** Reads `evaluation/results.json["threshold_global"]` and reports accuracy at that fixed threshold. This is the actual generalization test; tuning a threshold on Pins would just measure "can the model be tuned for Pins" instead.

Output `evaluation/results_pins.json` includes:
- `accuracy_at_lfw_threshold` (the headline number for generalization)
- `lfw_threshold_used`
- `pos_sim_mean`, `neg_sim_mean`, `spread` (for inspection)
- For reference / curiosity only: `accuracy_at_pins_tuned_threshold` and `pins_tuned_threshold` — clearly labeled as not the headline.

Applies only `_assert_distribution_sane` (spread, std, label balance). **Does not** call `_assert_threshold_sane` — Pins doesn't sweep a threshold of its own, so a threshold-bound check is meaningless here.

### Notebooks

`evaluation/evaluation_results.ipynb` already plots ROC, confusion matrix, etc. After the new eval lands the notebook re-executes against the new `results.json` and rerenders. No structural change.

## Data flow

- Phase 1: `manifest.parquet[train] → FaceDataset → RandomSampler → batch of (img, label) → forward+classifier → CE`.
- Phase 2: `manifest.parquet[train] → FaceDataset → PKSampler(32×4) → batch of (img, label) → forward → semi-hard triplet loss`.
- Eval: `pairs.txt → load_pairs_txt (no drop) → for each unique path: aligned-or-raw-fallback → batch embed → cosine sim → sanity check → 10-fold threshold sweep`.

## Error handling

- MTCNN alignment fails on a training image → already filtered at preprocess time; manifest only contains successful aligns. Phase 1/2 dataset never sees these.
- Aligned LFW image missing → fall back to center-cropped raw image via `_load_image_with_fallback`. Never `None`.
- Raw LFW image missing → setup bug; `_load_image_with_fallback` raises `AssertionError`. Not a runtime path.
- Semi-hard mining selects zero triplets in a batch → loss = `emb.sum() * 0.0`, n_triplets = 0; train loop skips `optimizer.step()`. A persistent run of zero-triplet batches is logged as a warning and aborts after 100 consecutive (signals broken sampler).
- Strict eval sanity assertion fails → `RuntimeError`, no `results.json` written. Final CLI exits non-zero.
- In-loop probe (strict=False) returns spread < 0.05 mid-training → just a log line; not an abort. The phase-1-end gate fires only after the full warmup is done.
- VM OOM or process kill → `phase1_end.pt`, `last.pt`, `best.pt` survive; resume by re-running phase 2 from `phase1_end.pt` (resume logic is a stretch goal, not required for first attempt).

## Testing strategy

Unit tests in `training_pipeline/tests/`:
- `test_loss.py::test_semi_hard_normalizes_input` — pass embeddings with non-unit norms, verify pairwise distances inside the loss are computed on unit-normalized vectors (no norm-scale escape).
- `test_loss.py::test_semi_hard_picks_correct_band` — handcrafted PK batch, verify selected negative is in `(d_ap, d_ap + margin)`.
- `test_loss.py::test_semi_hard_fallback` — no in-band negative, verify argmin-greater-than-d_ap fallback.
- `test_loss.py::test_semi_hard_no_valid_skips` — no negative satisfies `d_an > d_ap`, verify `(loss, n_triplets) = semi_hard_triplet_loss(...)` returns `n_triplets == 0` and `loss.backward()` is safe (no NaN, no graph error).
- `test_loss.py::test_semi_hard_returns_n_triplets` — verify return signature is `(Tensor, int)` and `n_triplets` matches the number of triplets actually used in the mean.
- `test_loss.py::test_softplus_smooth_gradient` — verify `softplus_loss` returns finite gradient on cases where hard hinge would return 0.
- `test_model.py::test_forward_unnormalized` — verify `model(x)` norms ≠ 1 (we removed L2-norm from training path).
- `test_model.py::test_embed_normalized` — verify `embed_normalized(x)` norms = 1 ± 1e-5.
- `test_eval_lfw.py::test_load_pairs_returns_all` — pass a pairs.txt with 6 pairs where 1 aligned image is missing, assert all 6 `LfwPair` objects still in return value.
- `test_eval_lfw.py::test_load_image_with_fallback` — deliberately missing aligned path, raw path present → returns tensor; missing both → raises.
- `test_eval_lfw.py::test_strict_collapse_fires` — feed collapsed embeddings with `strict=True`, assert `RuntimeError`. With `strict=False`, returns dict containing low spread.
- `test_eval_lfw.py::test_strict_label_leak_fires` — feed 95%-positive pairs with `strict=True`, assert `RuntimeError`.

End-to-end smoke (update existing `test_train_smoke`):
- 5 identities × 4 synthetic images, 1 phase-1 epoch + 1 phase-2 epoch, batch=8, no LFW probe.
- Assert phase-1 produces embeddings with non-trivial spread on the 20-sample synthetic set.
- Assert phase-2 doesn't collapse them back.

Manual VM verification (after training run):
1. `python -m evaluation.eval_lfw` — expect spread > 0.3, accuracy > 80%, threshold ∈ (0.0, 0.9).
2. `python -m evaluation.eval_pins` — expect cross-dataset accuracy > 70%.
3. `python /tmp/diagnose.py` — same script that originally caught the bug; expect clear spread between same-identity and different-identity cosine sim.
4. Spin up the FastAPI app, enroll two distinct faces, verify same+different — same should match, different should reject.

## Open questions

None — research already confirmed every choice. Hyperparameter ranges (lr, margin, phase split) are canonical from Hermans 2017 + MassFace.

## References

- Schroff et al. 2015 — FaceNet. Source of: semi-hard negative mining; L2-normalize embeddings during triplet training.
- Hermans et al. 2017 — In Defense of the Triplet Loss for Person Re-Identification. Source of: soft-margin variant `softplus(d_ap - d_an)`. Their batch-hard variant is where we got the collapse trap from when used cold.
- Wang et al. 2019 — MassFace. Source of: classification-softmax warmup → triplet handoff recipe; CASIA-only training to 98.3% LFW.
- Olivier Moindrot's blog — Triplet Loss and Online Triplet Mining in TensorFlow. Source of: explicit collapse warning + semi-hard mining pseudocode.
- Zhong et al. 2017 — Random Erasing Data Augmentation. Context for why we drop it from 160×160 face-crop training (erases identity regions).
