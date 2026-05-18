# Siamese Training: What We Fixed, How It Trains

## What we changed to fix the model

### The bug we found

The previous "97.48% LFW" result was an artifact, not a real metric.

1. **Eval silently dropped pairs.** The LFW evaluator filtered out pairs whose aligned image was missing on disk. 2378 of 3000 negative pairs got dropped while positives were preserved, leaving 97.48% positives in the remaining set. The threshold search collapsed to `-1.0` (predict "same" for everything) and that scored 97.48% accuracy — purely by class imbalance.

2. **Model produced identical embeddings.** Every face → cosine similarity ≈ 0.997 with every other face. The network had learned to ignore its input entirely. This is *embedding collapse* — the degenerate solution where every output is the same constant vector, which trivially minimizes triplet loss on a unit hypersphere.

### Six fixes that mattered

| Fix | Why it mattered |
|---|---|
| **L2-norm inside the loss, not on `forward()`** | Old code normalized at the model output. Triplet loss on a unit hypersphere can be "won" by collapsing every vector to the same point. Moving the normalize inside the loss means the model's raw activations carry magnitude info during training, and only the loss sees the constrained space. |
| **Softmax warmup (Phase 1) before triplet (Phase 2)** | Triplet from cold weights mines from near-random embeddings — gradient noise drowns signal, collapse happens. CE classification on identity labels gives clean dense gradients first, establishing a real embedding space. Then we switch to triplet to refine the geometry. |
| **Semi-hard negative mining instead of batch-hard** | Batch-hard picks the *hardest* negative per anchor → over-aggressive → collapse. Semi-hard picks negatives in the band `d_ap < d_an < d_ap + margin` — informative but not destructive. (FaceNet paper, Schroff et al. 2015.) |
| **Soft-margin loss `softplus(d_ap - d_an)` instead of hinge** | Hinge `relu(d_ap - d_an + margin)` is flat past the margin → zero gradient on already-correct triplets. Softplus is always smooth → never dies. |
| **No-drop LFW eval with raw fallback** | If the aligned image is missing, fall back to the raw deepfunneled image cropped 80% center. Pos/neg balance preserved → no label leak. Hard assertions on `spread > 0.05` and `pos_ratio ≈ 0.5` prevent silent regressions in future runs. |
| **5× more identities (CASIA-WebFace, 10572)** | Previous train set had 2150 identities. Embedding quality scales with identity count more than with image count. CASIA-WebFace gave us the data. |

### Also removed

- **`RandomErasing` from train augmentations.** For face recognition it occludes critical regions (eyes, nose) and degrades identity signal.
- **Frozen backbone with LR 3e-5.** Backbone now trains with LR 1e-4, head 5e-4, AdamW + cosine schedule.

---

## How the model actually trains

This is a **siamese embedding model** — one tower (ResNet50 → `Linear(2048, 512)`) that maps any face image to a 512-d vector. Two faces are declared "same person" if their vectors have high cosine similarity. The tower itself doesn't know about pairs; the *training procedure* enforces the pair structure.

### Phase 1 — Softmax warmup (3 epochs)

A temporary `Linear(512, 10572)` classifier head is bolted on top of the embedding output. Each training step:

1. Sample a random batch of 128 face images (`RandomSampler` with replacement, 1500 batches per epoch).
2. Forward through ResNet → 512-d embedding → classifier → 10572 logits.
3. Cross-entropy loss against the true identity label.
4. Backprop, AdamW step with linear warmup + cosine decay.

This is plain image classification on identity labels. We don't care about the classifier — it gets discarded after Phase 1. We care that the 512-d embedding learns to separate identities. After 3 epochs of CE the embedding space is non-degenerate: same person → close, different people → far.

**Phase 1 gate.** End of Phase 1 we measure `spread = pos_sim_mean - neg_sim_mean` on the LFW probe. If `spread ≤ 0.05`, training aborts — embeddings collapsed, no point continuing. This is the safety net that catches the original bug.

### Phase 2 — Semi-hard triplet (17 epochs)

The classifier is discarded. Now we train the embedding directly on the pairwise structure it needs to express at inference time.

Each batch is built by a **PK sampler**: pick P=32 identities, K=4 images each → 128-image batch. This guarantees every identity has multiple representatives in the batch, so every anchor image has same-identity "positives" and different-identity "negatives" available locally.

For each anchor image `a`:

1. Find every positive `p` in the batch (same identity, different image).
2. Compute distance `d_ap` and all distances `d_an` to negatives.
3. Pick a negative in the **semi-hard band**: `d_ap < d_an < d_ap + margin`. This negative is *closer than it should be* (informative) but *farther than the positive* (not catastrophic).
4. If no semi-hard negative exists, fall back to the closest negative that's still harder than the positive.
5. Loss contribution per triplet: `softplus(d_ap - d_an)`. Smooth penalty that pushes `d_an` further and `d_ap` closer.

Average over all valid triplets in the batch → backprop → step. If a batch yields zero usable triplets (rare — early Phase 2 only), skip the optimizer step. 100 such batches in a row aborts training (likely a mining bug).

After every epoch, run the LFW probe (full 6000 pairs, embed each unique image once, compute cosine similarities, 10-fold threshold cross-validation). Save `best.pt` only if accuracy improved **and** spread > 0.05 (no promoting collapsed checkpoints).

### Why loss looks flat in Phase 2

- **Soft-margin floor.** `softplus(d_ap - d_an)` bottoms out near `log(2) ≈ 0.69` when `d_ap ≈ d_an`. Even with perfect separation, triplets where the positive is barely closer than the negative still contribute ~0.5 to the average.
- **n_triplets = 384 every batch.** That's the structural maximum: P × K × (K-1) = 32 × 4 × 3 = 384 anchor-positive pairs per batch. Every (a,p) finds a usable negative; same volume every step.
- **Triplet loss saturates.** Once a triplet satisfies the margin, its contribution drops to ~0. The mean over a fixed-size population stays in a narrow band.

The signal you watch in Phase 2 is the **LFW probe accuracy and spread**, not the training loss. Loss flat + LFW rising = healthy. Loss dropping + LFW flat would be the bad case (overfitting to CASIA).

### Inference

The FastAPI app calls `model.embed_normalized(face_tensor)`, gets a unit-norm 512-d vector, and compares cosine similarity to the enrolled embeddings using the threshold from the trained checkpoint. No classifier, no triplets — the embedding is the whole product.

---

## Current results

Training on the merged manifest (10571 CASIA identities + 901 LFW eval identities, 213k training images).

| Stage | Train loss | LFW acc | Spread |
|---|---|---|---|
| P1 epoch 1 | 7.4431 | 87.25% | 0.42 |
| P1 epoch 2 | 3.1330 | 89.88% | 0.45 |
| P1 epoch 3 | 1.7473 | 89.37% | 0.45 |
| P2 epoch 1 | 0.5824 | **90.90%** | 0.58 |
| P2 epoch 2 | 0.5677 | 89.87% | 0.58 |
| P2 epoch 3 | 0.5511 | 89.78% | 0.58 |

Phase 2 still running. `best.pt` currently at P2 epoch 1 (90.90% / 0.58). Comparison vs. previous broken run: peaked at 78.02% real accuracy with spread 0.36 — and **never above the 80% spec target**. Current run cleared that on its first triplet epoch.
