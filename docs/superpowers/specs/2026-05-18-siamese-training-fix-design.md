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

  image → backbone → embedding head → (no L2 norm) → embedding ∈ R^512
                                                          ↓
                                      semi-hard triplet selection within PK batch
                                                          ↓
                                              softplus(d_ap - d_an)

Inference / eval:

  image → backbone → embedding head → F.normalize(out, p=2, dim=1) → unit-norm embedding
  cosine_sim(a, b) = a · b
```

## Components

### `training_pipeline/src/model.py`

- `FaceEmbedding.forward(x)` returns **un-normalized** 512-d vector. (Previously normalized.)
- New `FaceEmbedding.embed_normalized(x)`: forward + L2-normalize. Used by eval, by app inference, and inside the triplet loss when computing pairwise *cosine* distances for selection if we want angle-space mining; current plan uses Euclidean distance on un-normalized vectors.
- New module `ClassifierHead(nn.Module)`:
  - Single `nn.Linear(embedding_dim, n_identities)`, no bias.
  - Used only during Phase 1; instantiated separately from `FaceEmbedding` so it can be cleanly dropped at handoff.

### `training_pipeline/src/loss.py`

- Keep `_pairwise_dist` (correct, unchanged).
- New `semi_hard_triplet_loss(emb, labels, margin=0.3)`:
  - For each (anchor a, positive p) where `labels[a] == labels[p]` and `a != p`:
    1. Compute `d_ap = dist[a, p]`.
    2. Build mask over j ≠ a where `labels[j] != labels[a]` and `d_ap < dist[a, j] < d_ap + margin`.
    3. If mask non-empty → sample one negative uniformly.
    4. Elif any j with `dist[a, j] > d_ap` → take argmin of those (closest harder-than-positive negative).
    5. Else → skip this anchor.
  - Collected triplets pass through `softplus_loss(d_ap, d_an) = log(1 + exp(d_ap - d_an))`. Mean over selected triplets.
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
- `cfg.train.lr_backbone = 1.0e-4`, `cfg.train.lr_head = 5.0e-4`, `cfg.train.weight_decay = 5.0e-4`.
- `cfg.train.margin = 0.3` (used by semi-hard selection band, not by the loss hinge).

Phase 1 loop:
1. Build `FaceEmbedding` + `ClassifierHead(512, n_identities)` (n derived from training manifest).
2. AdamW over both with split learning rates.
3. Per batch: forward → classifier logits → CE → backward → step.
4. End of epoch: run LFW probe (full 6000 pairs, fallback to raw on MTCNN miss). Print spread, log to history.json.
5. After last warmup epoch: assert `spread > 0.05`. If not, abort training (collapse signal — something is wrong with data or env).
6. Save `checkpoints/phase1_end.pt`.

Phase 2 loop:
1. Drop classifier head (delete reference, free parameters from optimizer).
2. Re-init optimizer with backbone + embedding head only, same split LRs.
3. Switch sampler to PKSampler (P=32, K=4) with `batches_per_epoch=1500`.
4. Cosine LR schedule across phase-2 epochs only, with 500-step warmup at the start of phase 2.
5. Per batch: forward → semi-hard triplet loss → backward → step. Mixed precision.
6. End of epoch: LFW probe + history.json append. Update `checkpoints/best.pt` if this epoch's probe LFW > best so far.

After phase 2: write `checkpoints/last.pt`.

### `training_pipeline/src/eval_lfw.py`

- `load_pairs_txt(pairs_txt)`: returns **all 6000 pairs** with no filtering on disk presence.
- New `_load_image_with_fallback(aligned_path: Path, raw_path: Path)`:
  - If `aligned_path.exists()`: use it via the standard `eval_transform`.
  - Else: open `raw_path`, take center crop = 80% of `min(w, h)`, resize to 160×160, then apply `eval_transform`. Always succeeds.
- `_PairImgDataset.__init__(pairs)` accepts the pair list directly and builds `(aligned_path, raw_path)` for each unique image.
- `evaluate_lfw(...)` adds four sanity assertions before the threshold sweep, in this order:

```python
pos = sims[labels == 1]
neg = sims[labels == 0]
spread = pos.mean() - neg.mean()
assert spread > 0.05, f"COLLAPSED: pos={pos.mean():.3f}, neg={neg.mean():.3f}"
assert pos.std() > 0.01, f"COLLAPSED: pos std={pos.std():.4f} too tight"
assert 0.4 < labels.mean() < 0.6, f"LABEL LEAK: {labels.mean():.2%} positive"
# (after threshold sweep)
assert -0.5 < threshold_global < 0.95, f"THRESHOLD AT BOUND: {threshold_global}"
```

Any failure → `RuntimeError("eval sanity check failed: <msg>")`. The training script catches this exception and **does not** update `best.pt`.

### `evaluation/eval_lfw.py` (CLI)

Unchanged in structure; calls into `evaluate_lfw` which now enforces sanity. Output `results.json` adds `pos_sim_mean`, `pos_sim_std`, `neg_sim_mean`, `neg_sim_std`, `spread`, `n_pairs_total`, `n_pairs_used`, `n_aligned`, `n_raw_fallback`.

### `evaluation/eval_pins.py` (new)

Same structure as `eval_lfw.py`. Loads the already-downloaded Pins dataset, builds 1500 positive + 1500 negative pairs at fixed seed, evaluates using the same `evaluate_lfw` engine with `pairs` constructed for Pins. Same sanity assertions apply. Output `evaluation/results_pins.json`.

### Notebooks

`evaluation/evaluation_results.ipynb` already plots ROC, confusion matrix, etc. After the new eval lands the notebook re-executes against the new `results.json` and rerenders. No structural change.

## Data flow

- Phase 1: `manifest.parquet[train] → FaceDataset → RandomSampler → batch of (img, label) → forward+classifier → CE`.
- Phase 2: `manifest.parquet[train] → FaceDataset → PKSampler(32×4) → batch of (img, label) → forward → semi-hard triplet loss`.
- Eval: `pairs.txt → load_pairs_txt (no drop) → for each unique path: aligned-or-raw-fallback → batch embed → cosine sim → sanity check → 10-fold threshold sweep`.

## Error handling

- MTCNN alignment fails on a training image → already filtered at preprocess time; manifest only contains successful aligns. Phase 1/2 dataset never sees these.
- MTCNN alignment fails on an LFW image at eval time → falls back to center-cropped raw image. Never `None`.
- Eval sanity assertion fails → `RuntimeError`, no results written, training script logs and refuses to update `best.pt`.
- VM OOM or process kill → checkpoints `phase1_end.pt`, `last.pt`, `best.pt` survive; resume by re-running phase 2 from `phase1_end.pt` (resume logic is a stretch goal, not required for first attempt).

## Testing strategy

Unit tests in `training_pipeline/tests/`:
- `test_loss.py::test_semi_hard_picks_correct_band` — handcrafted PK batch, verify selected negative is in `(d_ap, d_ap + margin)`.
- `test_loss.py::test_semi_hard_fallback` — no in-band negative, verify argmin-greater-than-d_ap fallback.
- `test_loss.py::test_semi_hard_no_valid_skips` — no negative satisfies `d_an > d_ap`, verify anchor skipped without NaN.
- `test_loss.py::test_softplus_smooth_gradient` — verify `softplus_loss` returns finite gradient on cases where hard hinge would return 0.
- `test_model.py::test_forward_unnormalized` — verify `model(x)` norms ≠ 1 (we removed L2-norm from training path).
- `test_model.py::test_embed_normalized` — verify `embed_normalized(x)` norms = 1 ± 1e-5.
- `test_eval_lfw.py::test_load_pairs_no_drop` — pass a pairs.txt with 6 pairs where 1 aligned image is missing, assert all 6 pairs still in return value.
- `test_eval_lfw.py::test_load_image_with_fallback` — deliberately missing aligned path, raw path present → returns tensor.
- `test_eval_lfw.py::test_sanity_collapse_fires` — feed collapsed embeddings, assert `RuntimeError`.
- `test_eval_lfw.py::test_sanity_label_leak_fires` — feed 95%-positive pairs, assert `RuntimeError`.

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

- Schroff et al. 2015 — FaceNet (semi-hard triplet, softmax warmup).
- Hermans et al. 2017 — In Defense of the Triplet Loss for Person Re-Identification (batch-all + soft-margin; their batch-hard variant is where we got the collapse trap from).
- Wang et al. 2019 — MassFace (CASIA-only softmax-warmup + semi-hard, 98.3% LFW).
- Olivier Moindrot's blog — Triplet Loss and Online Triplet Mining in TensorFlow (collapse warning + semi-hard pseudocode).
- Random Erasing Data Augmentation (Zhong et al. 2017) — context for why we drop it from face-crop training.
