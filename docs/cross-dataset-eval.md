# Cross-Dataset Generalization Report

Eight standard verification benchmarks (InsightFace test suite) run against `application/models/best.pt`. All results in `evaluation/results_insightface_bench.json`.

Each benchmark uses the LFW-style protocol: 10-fold cross-validation on cosine similarity threshold, plus the same accuracy measured at our LFW-tuned `τ = 0.565` (no per-dataset retuning — this is the honest cross-dataset number).

## Results

| Benchmark | What it tests | acc (CV) | acc @ τ=0.565 | Spread | Δ vs LFW |
|---|---|---:|---:|---:|---:|
| **lfw**       | Standard verification (sanity baseline) | **95.03%** | 94.73% | 0.724 | — |
| **cfp_ff**    | Frontal vs frontal                       | 94.79% | 93.39% | 0.753 | −0.24 |
| **cfp_fp**    | Frontal vs **profile**                   | 88.27% | 79.16% | 0.565 | −6.76 |
| **agedb_30**  | Same person, 30-year age gap             | 76.03% | 66.08% | 0.362 | −19.00 |
| **cplfw**     | Cross-pose LFW                           | 78.10% | 76.40% | 0.377 | −16.93 |
| **calfw**     | Cross-age LFW                            | 78.03% | 76.38% | 0.387 | −17.00 |
| **sllfw**     | Similar-looking lookalikes (hard negs)   | 77.80% | 71.02% | 0.256 | −17.23 |
| **talfw**     | Transfer-attack LFW (adversarial)        | 70.13% | 65.18% | 0.289 | −24.90 |

## Reading the table

**Healthy signals**

- `lfw` and `cfp_ff` are nearly identical (95.03% / 94.79%). Both are clean frontal verification — our model handles the "easy" case well. Spread > 0.7 on both: positives and negatives are clearly separated.
- The CV-tuned accuracy and the fixed-`τ` accuracy are *close* on lfw / cfp_ff / cplfw / calfw — meaning τ = 0.565 transfers reasonably across datasets. The model isn't winning by exploiting a dataset-specific threshold.

**Clear weaknesses (in order of magnitude)**

1. **Adversarial robustness (talfw, −24.9):** Transfer-attack LFW perturbs frontal faces with imperceptible noise crafted to break recognition models. We drop 25 points. This is expected and consistent with the literature — Siamese embedding models without adversarial training are not robust to this attack. **Mitigation:** adversarial training or a separate liveness/anti-spoof model. Out of scope for the current project but a real failure mode in adversarial settings.

2. **Age robustness (agedb_30, −19.0 ; calfw, −17.0):** Two age benchmarks both drop ~18 points. This is the **clearest training-data weakness.** CASIA-WebFace has shallow temporal coverage per identity — most images per person are from the same era. Without same-person-across-decades examples, the embedding learns to separate identities partly via age-stable features (face shape) but also via age-correlated features (skin tone, contour). Across a 30-year gap those correlated features drift, and the embedding follows.

3. **Hard negatives (sllfw, −17.2):** Confusable lookalikes. Spread collapses to 0.256 here (vs 0.72 on lfw). Our soft-margin semi-hard mining provides only mild pressure on near-boundary cases — once positives are inside the margin, the loss goes to ~0. Hard-negative-only mining (or an angular margin loss like ArcFace) would push the negative band further.

4. **Pose (cfp_fp, −6.8 ; cplfw, −16.9):** Profile faces are harder than frontals, but only catastrophically so on cplfw which is more extreme. CFP-FP's 88.3% suggests the model handles moderate yaw fine. Frontal training bias is the cause.

## Is the model overfitting?

**No, not in the classical sense.**

If this were CASIA-memorization overfit, you'd see LFW eval rise during training while everything else stayed flat or fell. Instead the picture is:

- LFW + CFP-FF strong (95%) — the model genuinely learned identity features.
- The weaknesses (age, lookalikes, extreme pose, adversarial) are **failure modes of the recipe and training data**, not memorization. A model that overfits to CASIA wouldn't generalize to LFW either.

What we have is the standard FaceNet/triplet-recipe profile: solid on the in-distribution case, vulnerable on the out-of-distribution axes the training data doesn't cover. To improve any specific axis you'd add data covering that axis (age progressions, mirror augmentations for pose, hard-mined negatives for lookalikes) or change the loss (ArcFace adds a large angular margin that helps all three).

## Recommended operating points

- **For the demo app**: τ = 0.5 (gives 89%+ acceptance for real users, occasional false matches that the demo can tolerate)
- **For accuracy benchmarking**: τ = 0.565 (matches our LFW eval, comparable cross-dataset numbers)
- **For security-sensitive use**: τ = 0.70+ — but at that point you'd want anti-spoof and adversarial defenses first; raising τ alone trades away usability for a problem the model can't fix.
