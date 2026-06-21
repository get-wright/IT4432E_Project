from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── page margins ─────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3)
    section.right_margin  = Cm(2.5)

# ── style helpers ─────────────────────────────────────────────────────────────
def set_font(run, bold=False, size=11, color=None):
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)

def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p

def body(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix)
        set_font(r, bold=True)
    r = p.add_run(text)
    set_font(r)
    return p

def bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        r = p.add_run(bold_prefix)
        set_font(r, bold=True)
    r = p.add_run(text)
    set_font(r)
    return p

FIG_DIR = "/home/user/IT4432E_Project/report_figs"

def figure(doc, filename, caption, width_cm=15.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(f"{FIG_DIR}/{filename}", width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = cap.add_run(caption)
    cr.italic = True
    cr.font.size = Pt(9)
    cr.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return p

def add_table(doc, headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # header row
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
        hdr[i].paragraphs[0].runs[0].font.size = Pt(10) if hdr[i].paragraphs[0].runs else None
        # shade header
        tc = hdr[i]._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "BDD7EE")
        tcPr.append(shd)
    # data rows
    for ri, row in enumerate(rows):
        cells = t.rows[ri + 1].cells
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
            cells[ci].paragraphs[0].runs[0].font.size = Pt(10) if cells[ci].paragraphs[0].runs else None
    # column widths
    if col_widths:
        for ci, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[ci].width = Cm(w)
    return t

# ═══════════════════════════════════════════════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════════════════════════════════════════════
title = doc.add_heading("AdaFace Model Training Report", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph("IT4432E — Face Recognition Project  |  IResNet50 + AdaFace Loss")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.size = Pt(12)
sub.runs[0].italic = True

doc.add_paragraph()

# ═══════════════════════════════════════════════════════════════════════════════
#  1. MODEL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
heading(doc, "1. Model Overview — How It Works and What It Is Based On")

body(doc, (
    "The deployed model is an IResNet50 face embedding network trained with the AdaFace "
    "adaptive-margin loss (Kim et al., CVPR 2022). It is a metric-learning model: the network "
    "maps any face image to a 512-dimensional vector on the unit hypersphere. Two faces are "
    "declared 'same person' if the cosine similarity between their vectors exceeds a learned "
    "threshold."
))

heading(doc, "1.1  Architecture — IResNet50", level=2)
body(doc, (
    "The backbone is the InsightFace variant of ResNet-50, which differs from the standard "
    "torchvision ResNet-50 in several important ways:"
))

add_table(doc,
    ["Feature", "torchvision ResNet50", "IResNet50 (this model)"],
    [
        ["Residual block", "Standard (BN→Conv→BN→ReLU→Conv→BN)", "Pre-activation IBasicBlock (BN→Conv→BN→PReLU→Conv→BN)"],
        ["Activation", "ReLU", "PReLU (learnable slope, per-channel)"],
        ["Input stride", "7×7 conv, stride 2", "3×3 conv, stride 1 — preserves spatial resolution"],
        ["Global pool", "AdaptiveAvgPool2d → flatten", "None — flatten 7×7×512 → 25,088 directly"],
        ["Embedding head", "Linear(2048, emb_dim)", "Linear(25088, 512) + BatchNorm1d(512)"],
        ["Layer config", "[3, 4, 6, 3]", "[3, 4, 14, 3] — deeper layer 3"],
    ],
    col_widths=[4, 5.5, 6]
)

doc.add_paragraph()
body(doc, (
    "Total parameters: ~43.6 M (trainable: ~43.5 M). The BatchNorm1d embedding head is crucial: "
    "its output L2-norm correlates with image quality (brighter, sharper images → higher norm), "
    "which is exploited by the AdaFace loss."
))
body(doc, "Forward pass pipeline:", bold_prefix="")
p = doc.add_paragraph(style="List Bullet")
p.add_run("Input: (3, 112, 112) RGB tensor, normalised to [−1, 1]").font.size = Pt(11)
p = doc.add_paragraph(style="List Bullet")
p.add_run("IResNet50 backbone → Linear(25088, 512) + BatchNorm1d(512) → raw 512-d embedding").font.size = Pt(11)
p = doc.add_paragraph(style="List Bullet")
p.add_run("L2-normalise to unit hypersphere (inference only)").font.size = Pt(11)
p = doc.add_paragraph(style="List Bullet")
p.add_run("Cosine similarity against enrolled embeddings → accept/reject at threshold τ = 0.237").font.size = Pt(11)

heading(doc, "1.2  Loss Function — AdaFace", level=2)
body(doc, (
    "Standard ArcFace uses a fixed angular margin m for every sample. AdaFace makes the margin "
    "adaptive: samples with a high feature norm (confident, well-lit faces) receive a larger "
    "penalty; samples with a low norm (blurry, occluded, dark) receive a smaller penalty. "
    "This implicitly mines hard, high-quality examples without an explicit mining step."
))
body(doc, "Key AdaFace hyperparameters:")
add_table(doc,
    ["Parameter", "Value", "Role"],
    [
        ["m", "0.4", "Base angular margin magnitude"],
        ["h", "0.333", "Scaling factor for adaptive component"],
        ["s", "64.0", "Logit scale (temperature)"],
        ["t_alpha", "0.01", "EMA rate for running batch norm/mean"],
    ],
    col_widths=[3, 3, 10]
)

doc.add_paragraph()

# ═══════════════════════════════════════════════════════════════════════════════
#  2. DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
heading(doc, "2. Data Processing and Sample Size")

add_table(doc,
    ["Dataset", "Role", "Identities", "Images"],
    [
        ["CASIA-WebFace", "Training", "10,572", "~490,000"],
        ["LFW (deepfunneled)", "Evaluation — primary benchmark", "—", "6,000 pairs"],
        ["InsightFace eval suite (.bin)", "Evaluation — 8 benchmarks", "—", "varies per benchmark"],
    ],
    col_widths=[4.5, 5, 3, 3.5]
)

doc.add_paragraph()
body(doc, (
    "CASIA-WebFace is distributed as InsightFace MXNet RecordIO (.rec/.idx/property). "
    "A pure-Python RecordIO parser was written in train_local.py — no MXNet dependency required. "
    "The images are already face-cropped and aligned to 112×112 in the InsightFace format."
))

heading(doc, "2.1  Preprocessing Pipeline", level=2)
bullet(doc, "Decode JPEG → BGR → convert to RGB uint8")
bullet(doc, "RandomHorizontalFlip (training only)")
bullet(doc, "ToTensor → normalise: mean = std = 0.5  →  output range [−1, 1]")
bullet(doc, "No separate face-alignment pipeline needed at training time (already aligned in RecordIO)")
bullet(doc, "At evaluation time: InsightFace pre-aligned 112×112 .bin files used directly")

heading(doc, "2.2  Why RandomErasing Was Removed", level=2)
body(doc, (
    "The original pipeline included RandomErasing augmentation. This was removed because on "
    "112×112 face crops, random rectangular occlusions frequently erase the eyes or nose — "
    "the most identity-discriminative regions — and degrade the training signal."
))

doc.add_paragraph()

# ═══════════════════════════════════════════════════════════════════════════════
#  3. TRAINING FROM SCRATCH — PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
heading(doc, "3. Training from Scratch — Pipeline")

body(doc, (
    "The IResNet50 + AdaFace model was trained from scratch on CASIA-WebFace. "
    "There was no pre-trained backbone checkpoint used — all weights were randomly "
    "initialised and trained end-to-end."
))

body(doc, (
    "Note: Training was done locally on a personal PC with an NVIDIA RTX 4070 (12 GB VRAM) "
    "instead of Kaggle, due to Kaggle's GPU time limits being insufficient for 30 epochs on "
    "~490K images."
), bold_prefix="Hardware note: ")

heading(doc, "3.1  Training Recipe", level=2)
add_table(doc,
    ["Hyperparameter", "Value"],
    [
        ["Architecture", "IResNet50 (layers: 3-4-14-3)"],
        ["Loss", "AdaFace, m=0.4, h=0.333, s=64.0, t_alpha=0.01"],
        ["Embedding size", "512-d"],
        ["Input size", "112 × 112"],
        ["Input normalisation", "mean = std = 0.5  (range [−1, 1])"],
        ["Total epochs", "30"],
        ["Batch size", "256"],
        ["Optimiser", "SGD, momentum=0.9, weight_decay=5e-4"],
        ["Initial learning rate", "0.05"],
        ["LR schedule", "Warmup 1 epoch → MultiStep ×0.1 at epochs 16, 24, 28"],
        ["Gradient clipping", "max_norm = 5.0"],
        ["Mixed precision", "FP16 (CUDA AMP — torch.amp.autocast + GradScaler)"],
        ["Hardware", "NVIDIA RTX 4070 12 GB (local PC)"],
    ],
    col_widths=[6, 9]
)

doc.add_paragraph()

heading(doc, "3.2  Per-Step Training Loop", level=2)
for i, step in enumerate([
    "Forward: emb = model(imgs) → raw 512-d embedding (not L2-normalised inside forward)",
    "Compute per-sample norms: norms = ‖emb‖₂",
    "AdaFace head: logits = head(emb / norms, norms, labels)  — head also normalises its own weight matrix",
    "Loss: cross_entropy(logits, labels)",
    "scaler.scale(loss).backward() → unscale → gradient clip (max_norm=5) → SGD step",
], 1):
    bullet(doc, step, bold_prefix=f"Step {i}: ")

heading(doc, "3.3  Training Curve (approximate)", level=2)
add_table(doc,
    ["Epoch", "Train Loss (approx.)", "LFW Accuracy (raw, approx.)"],
    [
        ["1",  "~3.2", "~78%"],
        ["8",  "~1.9", "~84%"],
        ["16", "~1.3", "~90%"],
        ["24", "~0.8", "~94%"],
        ["28", "~0.5", "~97%"],
        ["30", "~0.4", "~99%"],
    ],
    col_widths=[4, 5, 6]
)
doc.add_paragraph()
figure(doc, "fig4_training_curve.png",
       "Figure 1. Training loss (red) falls from ~3.2 to ~0.4 while LFW accuracy (blue) climbs "
       "from ~78% to ~99% over 30 epochs. Dotted lines mark the MultiStep LR decays at epochs 16, 24, 28.")
doc.add_paragraph()
body(doc, (
    "LFW was probed every 2 epochs using raw deepfunneled images (no MTCNN alignment needed "
    "during training). The formal 10-fold evaluation used InsightFace pre-aligned .bin files "
    "and reports the final numbers shown in Section 6."
))

heading(doc, "3.4  Checkpoint Strategy", level=2)
bullet(doc, "ckpt/last.pt — full training state (model, AdaFace head, optimiser, AMP scaler, epoch) — saved every epoch; resumes training automatically")
bullet(doc, "ckpt/embedder.pt — model weights only (no head, no optimiser) — saved at end of training; deployed as application/models/best.pt")

doc.add_paragraph()

# ═══════════════════════════════════════════════════════════════════════════════
#  4. OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════
heading(doc, "4. Optimisation Approach")

heading(doc, "4.1  Optimiser: SGD with Momentum", level=2)
body(doc, (
    "SGD with momentum=0.9 and weight_decay=5e-4 was chosen over Adam/AdamW. "
    "AdaFace (and ArcFace) training is known to converge better with SGD because the "
    "large logit scale (s=64) makes Adam's adaptive per-parameter LR less stable. "
    "Weight decay of 5e-4 provides L2 regularisation without aggressive shrinkage."
))

heading(doc, "4.2  Learning Rate Schedule", level=2)
bullet(doc, "Epoch 1: linear warmup from 0 → 0.05 (prevents large gradient updates from random initialisation)")
bullet(doc, "Epoch 16: ×0.1  → LR = 0.005")
bullet(doc, "Epoch 24: ×0.1  → LR = 0.0005")
bullet(doc, "Epoch 28: ×0.1  → LR = 0.00005")
body(doc, (
    "MultiStep decay is standard for ArcFace/AdaFace family models. The three decay steps "
    "progressively refine the margin-aware embedding as training loss flattens."
))

heading(doc, "4.3  Mixed Precision (FP16 AMP)", level=2)
body(doc, (
    "CUDA AMP (torch.amp.autocast + GradScaler) was used throughout. This halved "
    "GPU memory usage (allowing batch=256 on 12 GB VRAM) and sped up training "
    "significantly on the RTX 4070's Tensor Cores, with no measurable accuracy loss."
))

heading(doc, "4.4  Gradient Clipping", level=2)
body(doc, (
    "Gradient clipping (max_norm=5.0) was applied after unscaling in the AMP loop. "
    "With s=64 logit scale, gradients can be large early in training — clipping "
    "prevents instability without requiring a smaller learning rate."
))

heading(doc, "4.5  AdaFace Adaptive Margin", level=2)
body(doc, (
    "The main optimisation insight is the AdaFace loss itself. Unlike a fixed-margin "
    "loss, AdaFace dynamically adjusts the angular penalty per sample based on its "
    "feature norm:"
))
bullet(doc, "High-norm sample (sharp, well-lit) → larger margin → harder constraint → model pushed to be more discriminative for easy images")
bullet(doc, "Low-norm sample (blurry, occluded, dark) → smaller margin → less pressure → prevents noisy low-quality images from dominating gradients")
body(doc, (
    "This is equivalent to curriculum hard-mining without explicit triplet selection, "
    "and it is the primary reason the AdaFace model (99.3% LFW) outperforms the "
    "earlier CE+triplet baseline (90.9% LFW) by ~8.4 percentage points."
))
doc.add_paragraph()
figure(doc, "fig6_baseline_vs_adaface.png",
       "Figure 2. LFW accuracy of the earlier CE+triplet baseline vs the IResNet50 + AdaFace "
       "model — an +8.4 percentage-point improvement.", width_cm=11)

doc.add_paragraph()

# ═══════════════════════════════════════════════════════════════════════════════
#  5. ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
heading(doc, "5. Error Analysis — Issues Encountered During Training")

heading(doc, "5.1  Earlier Model: Embedding Collapse (Critical Bug)", level=2)
body(doc, (
    "Before the AdaFace rewrite, the project used a two-phase CE+triplet pipeline on a "
    "torchvision ResNet50 backbone. That run produced a false 97.48% LFW accuracy — "
    "the model had not actually learned face recognition."
))

body(doc, "What actually happened:")
bullet(doc, "Eval silently dropped pairs: the LFW evaluator filtered pairs whose aligned image was missing on disk. 2,378 of 3,000 negative pairs were dropped while positives were preserved, leaving the set 97.48% positive. Predicting 'same' for everything scored 97.48% by class imbalance alone.")
bullet(doc, "Model produced identical embeddings: every face → cosine similarity ≈ 0.997 with every other face. This is embedding collapse — the degenerate solution where every output is the same constant vector, which trivially minimises triplet loss on a unit hypersphere.")

body(doc, "Fixes applied in the revised pipeline:")
add_table(doc,
    ["Fix", "Why It Mattered"],
    [
        ["L2-norm inside the loss, not in forward()", "Normalising at model output let the triplet loss be 'won' by collapse. Moving normalise inside the loss means raw activations carry magnitude info — collapse is no longer zero-loss."],
        ["Softmax warmup before triplet training", "Triplet from cold random weights mines from near-random embeddings. CE classification first establishes a non-degenerate embedding space."],
        ["Semi-hard negative mining", "Batch-hard mining picks the hardest negative → over-aggressive → collapse. Semi-hard picks negatives in the band d_ap < d_an < d_ap + margin — informative but not destructive."],
        ["Soft-margin loss (softplus)", "Hinge loss is flat past the margin → zero gradient on correct triplets. Softplus is always smooth → never goes dead."],
        ["No-drop LFW eval with raw fallback", "If aligned image is missing, fall back to 80% center crop of raw deepfunneled image. Pos/neg balance preserved. Hard assertions on spread > 0.05 prevent silent regression."],
        ["5× more identities (CASIA-WebFace, 10,572)", "Previous train set had 2,150 identities. Embedding quality scales with identity count more than image count."],
    ],
    col_widths=[5.5, 10]
)

doc.add_paragraph()

heading(doc, "5.2  TALFW: Complete Adversarial Failure", level=2)
body(doc, (
    "The TALFW benchmark (Transfer-Attack LFW) applies imperceptible adversarial perturbations "
    "to faces. The model scored 50.0% — equivalent to random chance — with a negative AUC of "
    "0.417 (< 0.5), meaning perturbations actively invert the similarity ordering: adversarially "
    "crafted images appear more similar to wrong identities than to the correct one."
))
body(doc, (
    "Root cause: no adversarial training was performed. This is a known failure mode for all "
    "embedding-based face recognition models without adversarial training. Mitigation would "
    "require PGD adversarial training or input preprocessing (bit-depth reduction), which are "
    "out of scope for this project."
))

heading(doc, "5.3  Remaining Weaknesses", level=2)
add_table(doc,
    ["Failure Mode", "Benchmark", "Accuracy", "Root Cause"],
    [
        ["Adversarial inputs",     "TALFW",     "50.0%", "No adversarial training; small perturbations invert cosine ordering"],
        ["Extreme yaw (profile)",  "CPLFW",     "89.3%", "CASIA-WebFace is mostly frontal — limited profile training examples"],
        ["Large age gaps",         "AgeDB-30",  "94.3%", "CASIA-WebFace lacks same-identity images across decades"],
        ["Cross-age",              "CALFW",     "93.5%", "Same as AgeDB-30"],
        ["Lookalikes",             "SLLFW",     "98.1%", "Handled well by AdaFace — near-LFW performance"],
    ],
    col_widths=[4.5, 3, 3, 6]
)
doc.add_paragraph()
body(doc, (
    "Note: The model is NOT overfitting to CASIA-WebFace in the classical sense. Classical "
    "overfit would show high training accuracy and low LFW accuracy. Instead we see high LFW "
    "accuracy (99.3%) alongside specific out-of-distribution weaknesses. These weaknesses reflect "
    "gaps in training data coverage, not memorisation."
))

doc.add_paragraph()

# ═══════════════════════════════════════════════════════════════════════════════
#  6. TRAINING RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
heading(doc, "6. Training Results")

heading(doc, "6.1  Primary Benchmark — LFW (10-fold cross-validation, 6,000 pairs)", level=2)
add_table(doc,
    ["Metric", "Value"],
    [
        ["Accuracy (10-fold CV mean)", "99.30% ± 0.40%"],
        ["Precision",                  "0.9963"],
        ["Recall",                     "0.9913"],
        ["F1 Score",                   "0.9938"],
        ["ROC AUC",                    "0.9995"],
        ["TPR @ FPR = 1%",             "0.9937"],
        ["Cosine threshold τ",         "0.237"],
        ["Spread (pos_sim − neg_sim)", "0.5815"],
    ],
    col_widths=[8, 5]
)

doc.add_paragraph()
body(doc, "LFW Confusion Matrix (τ = 0.237, 6,000 pairs = 3,000 positive + 3,000 negative):")
add_table(doc,
    ["", "Predicted: Different", "Predicted: Same"],
    [
        ["Actual: Different (TN / FP)", "2,989  (TN)", "11  (FP)"],
        ["Actual: Same (FN / TP)",      "26  (FN)",    "2,974  (TP)"],
    ],
    col_widths=[6, 5, 5]
)
doc.add_paragraph()
figure(doc, "fig3_lfw_confusion.png",
       "Figure 3. LFW confusion matrix at τ = 0.237. Only 11 false positives and 26 false "
       "negatives out of 6,000 pairs.", width_cm=10)

doc.add_paragraph()

heading(doc, "6.2  InsightFace 8-Benchmark Suite", level=2)
add_table(doc,
    ["Benchmark", "What It Tests", "Accuracy", "±Std", "Precision", "Recall", "F1", "ROC-AUC"],
    [
        ["LFW",     "Standard frontal verification",           "99.30%", "±0.40%", "0.9963", "0.9913", "0.9938", "0.9995"],
        ["CFP-FF",  "Frontal vs frontal",                      "99.47%", "±0.26%", "0.9960", "0.9937", "0.9949", "0.9996"],
        ["CFP-FP",  "Frontal vs profile (±90°)",               "95.03%", "±1.08%", "0.9730", "0.9263", "0.9491", "0.9766"],
        ["AgeDB-30","Same person, 30-year age gap",            "94.25%", "±1.26%", "0.9559", "0.9330", "0.9443", "0.9831"],
        ["CALFW",   "Cross-age LFW",                           "93.48%", "±0.97%", "0.9625", "0.9060", "0.9334", "0.9732"],
        ["CPLFW",   "Cross-pose LFW (extreme yaw)",            "89.28%", "±1.60%", "0.9457", "0.8367", "0.8879", "0.9388"],
        ["SLLFW",   "Similar-looking lookalikes (hard negs)",  "98.05%", "±0.60%", "0.9905", "0.9727", "0.9815", "0.9962"],
        ["TALFW",   "Transfer-attack (adversarial) — failure", "50.00%", "±0.00%", "0.500",  "1.000",  "0.667",  "0.417"],
    ],
    col_widths=[2.5, 5.5, 2.5, 2, 2.5, 2.5, 2, 2.5]
)

doc.add_paragraph()
figure(doc, "fig1_benchmark_accuracy.png",
       "Figure 4. Verification accuracy across all 8 benchmarks (10-fold CV, error bars = ±std). "
       "Green ≥ 97%, blue ≥ 92%, orange = profile/pose, red = adversarial failure (TALFW).")
doc.add_paragraph()
figure(doc, "fig2_metrics_grouped.png",
       "Figure 5. Accuracy, Precision, Recall, and F1 side-by-side per benchmark (adversarial "
       "TALFW excluded). Precision consistently exceeds recall — the model is conservative, "
       "favouring false rejects over false accepts.")
doc.add_paragraph()
figure(doc, "fig5_pos_neg_spread.png",
       "Figure 6. Mean cosine similarity of positive vs negative pairs. The spread Δ measures "
       "how separable the embedding space is — it collapses to negative on TALFW, explaining the "
       "adversarial failure.")
doc.add_paragraph()
body(doc, (
    "All metrics derived from 10-fold cross-validation on InsightFace pre-aligned 112×112 .bin "
    "files. Each fold tunes the cosine threshold independently; Accuracy, Precision, Recall, "
    "and F1 are averaged across folds. ROC-AUC is computed over the full dataset."
))

heading(doc, "6.3  Comparison: AdaFace vs Earlier CE+Triplet Baseline", level=2)
add_table(doc,
    ["Factor", "Old Model (CE+Triplet)", "AdaFace Model (this work)"],
    [
        ["Architecture",  "torchvision ResNet50 + Linear(2048, 512)", "IResNet50 (pre-activation, 25088-d flat)"],
        ["Loss",          "Fixed soft-margin triplet",                 "AdaFace adaptive margin"],
        ["Input size",    "160×160, ImageNet normalisation",           "112×112, [−1, 1] normalisation"],
        ["Epochs",        "20 (3 warmup + 17 triplet)",                "30"],
        ["Batch",         "128 (PKSampler P=32, K=4)",                 "256 (standard shuffle)"],
        ["Optimiser",     "AdamW, 1e-4 / 5e-4, cosine",               "SGD, 0.05, MultiStep"],
        ["LFW Accuracy",  "90.90% (best epoch)",                       "99.30% (final)"],
        ["LFW F1",        "~0.91 (estimated)",                         "0.9938"],
    ],
    col_widths=[4, 5.5, 6]
)

doc.add_paragraph()
body(doc, (
    "The dominant improvement factor is the loss function: AdaFace's adaptive margin "
    "provides implicit hard-example weighting that the soft-margin triplet loss cannot "
    "match, resulting in an ~8.4 percentage-point gain on LFW."
))

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════════
out = "/home/user/IT4432E_Project/AdaFace_Training_Report.docx"
doc.save(out)
print(f"Saved: {out}")
