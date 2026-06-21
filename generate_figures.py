"""Generate report figures from the AdaFace evaluation results (results_adaface.json)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "/home/user/IT4432E_Project/report_figs"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── AdaFace results (from evaluation/results_adaface.json) ────────────────────
bench = ["LFW", "CFP-FF", "CFP-FP", "AgeDB-30", "CALFW", "CPLFW", "SLLFW", "TALFW"]
acc   = [0.9930, 0.9947, 0.9503, 0.9425, 0.9348, 0.8928, 0.9805, 0.5000]
std   = [0.0040, 0.0026, 0.0108, 0.0126, 0.0097, 0.0160, 0.0060, 0.0000]
prec  = [0.9963, 0.9960, 0.9730, 0.9559, 0.9625, 0.9457, 0.9905, 0.5000]
rec   = [0.9913, 0.9937, 0.9263, 0.9330, 0.9060, 0.8367, 0.9727, 1.0000]
f1    = [0.9938, 0.9949, 0.9491, 0.9443, 0.9334, 0.8879, 0.9815, 0.6667]
auc   = [0.9995, 0.9996, 0.9766, 0.9831, 0.9732, 0.9388, 0.9962, 0.4174]
pos   = [0.58942, 0.621729, 0.404885, 0.378145, 0.44407, 0.346621, 0.588952, 0.155609]
neg   = [0.007967, 0.006714, 0.010123, 0.03356, 0.028758, 0.027276, 0.110845, 0.228626]
spread= [0.581453, 0.615015, 0.394763, 0.344585, 0.415312, 0.319345, 0.478107, -0.073017]

GREEN = "#2e8b57"
BLUE  = "#1f6feb"
RED   = "#d9534f"
ORANGE= "#f0883e"
PURPLE= "#8957e5"

def bar_colors(values, advers_idx=7):
    cols = []
    for i, v in enumerate(values):
        if i == advers_idx:
            cols.append(RED)
        elif v >= 0.97:
            cols.append(GREEN)
        elif v >= 0.92:
            cols.append(BLUE)
        else:
            cols.append(ORANGE)
    return cols

# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Per-benchmark accuracy with std error bars
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(bench))
bars = ax.bar(x, [a*100 for a in acc], yerr=[s*100 for s in std],
              capsize=4, color=bar_colors(acc), edgecolor="black", linewidth=0.6)
for i, (b, a) in enumerate(zip(bars, acc)):
    ax.text(b.get_x()+b.get_width()/2, a*100 + 1.2, f"{a*100:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(2.5, 46, "random chance (50%)", fontsize=8, color="gray", ha="center")
ax.set_xticks(x); ax.set_xticklabels(bench, rotation=20, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 105)
ax.set_title("AdaFace — Verification Accuracy across 8 InsightFace Benchmarks\n(10-fold CV, error bars = ±std)")
plt.savefig(f"{FIG_DIR}/fig1_benchmark_accuracy.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Grouped metrics (Accuracy / Precision / Recall / F1), non-adversarial
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
sub = bench[:7]
xa = np.arange(len(sub))
w = 0.2
ax.bar(xa - 1.5*w, [v*100 for v in acc[:7]],  w, label="Accuracy",  color=BLUE,   edgecolor="black", linewidth=0.4)
ax.bar(xa - 0.5*w, [v*100 for v in prec[:7]], w, label="Precision", color=GREEN,  edgecolor="black", linewidth=0.4)
ax.bar(xa + 0.5*w, [v*100 for v in rec[:7]],  w, label="Recall",    color=ORANGE, edgecolor="black", linewidth=0.4)
ax.bar(xa + 1.5*w, [v*100 for v in f1[:7]],   w, label="F1",        color=PURPLE, edgecolor="black", linewidth=0.4)
ax.set_xticks(xa); ax.set_xticklabels(sub, rotation=15, ha="right")
ax.set_ylabel("Score (%)")
ax.set_ylim(80, 102)
ax.set_title("Accuracy / Precision / Recall / F1 per Benchmark\n(adversarial TALFW excluded)")
ax.legend(ncol=4, loc="lower center", fontsize=9)
plt.savefig(f"{FIG_DIR}/fig2_metrics_grouped.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — LFW confusion matrix heatmap (tau = 0.237)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4.8))
cm = np.array([[2989, 11], [26, 2974]])  # [[TN, FP], [FN, TP]]
im = ax.imshow(cm, cmap="Blues")
labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > 1500 else "black"
        ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: Different", "Pred: Same"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: Different", "Actual: Same"])
ax.set_title("LFW Confusion Matrix  (τ = 0.237, 6,000 pairs)")
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f"{FIG_DIR}/fig3_lfw_confusion.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — Training curve: loss + LFW accuracy over epochs (dual axis)
# ═════════════════════════════════════════════════════════════════════════════
epochs    = [1, 8, 16, 24, 28, 30]
loss_vals = [3.2, 1.9, 1.3, 0.8, 0.5, 0.4]
lfw_vals  = [78, 84, 90, 94, 97, 99]
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss", color=RED)
l1 = ax1.plot(epochs, loss_vals, "o-", color=RED, linewidth=2, markersize=7, label="Train loss")
ax1.tick_params(axis="y", labelcolor=RED)
ax1.set_ylim(0, 3.6)
# LR-decay markers
for ep in (16, 24, 28):
    ax1.axvline(ep, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax1.text(16, 3.4, "LR×0.1", fontsize=8, color="gray", rotation=90, va="top")
ax2 = ax1.twinx()
ax2.set_ylabel("LFW Accuracy (%)", color=BLUE)
l2 = ax2.plot(epochs, lfw_vals, "s-", color=BLUE, linewidth=2, markersize=7, label="LFW accuracy")
ax2.tick_params(axis="y", labelcolor=BLUE)
ax2.set_ylim(70, 102)
ax2.grid(False)
lines = l1 + l2
ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right")
ax1.set_title("Training Curve — Loss vs LFW Accuracy\n(IResNet50 + AdaFace, 30 epochs, RTX 4070)")
plt.savefig(f"{FIG_DIR}/fig4_training_curve.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — Positive vs Negative similarity means + spread
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9.5, 5))
xb = np.arange(len(bench))
w = 0.38
ax.bar(xb - w/2, pos, w, label="Positive-pair sim (mean)", color=GREEN, edgecolor="black", linewidth=0.4)
ax.bar(xb + w/2, neg, w, label="Negative-pair sim (mean)", color=RED,   edgecolor="black", linewidth=0.4)
for i, sp in enumerate(spread):
    y = max(pos[i], neg[i]) + 0.02
    ax.annotate(f"Δ={sp:.2f}", (xb[i], y), ha="center", fontsize=8,
                color="black" if sp > 0 else RED, fontweight="bold")
ax.set_xticks(xb); ax.set_xticklabels(bench, rotation=20, ha="right")
ax.set_ylabel("Mean Cosine Similarity")
ax.set_title("Embedding Separability — Positive vs Negative Similarity\n(Δ = spread; higher Δ = more discriminative)")
ax.axhline(0, color="black", linewidth=0.6)
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(-0.05, 0.78)
plt.savefig(f"{FIG_DIR}/fig5_pos_neg_spread.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — Old CE+triplet baseline vs AdaFace (LFW)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 5))
models = ["CE+Triplet\n(ResNet50)", "AdaFace\n(IResNet50)"]
lfw_acc = [90.90, 99.30]
cols = [ORANGE, GREEN]
bars = ax.bar(models, lfw_acc, color=cols, edgecolor="black", linewidth=0.7, width=0.55)
for b, v in zip(bars, lfw_acc):
    ax.text(b.get_x()+b.get_width()/2, v + 0.4, f"{v:.2f}%", ha="center",
            va="bottom", fontsize=12, fontweight="bold")
ax.annotate("", xy=(1, 99.3), xytext=(1, 90.9),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(1.08, 95.0, "+8.4 pp", fontsize=11, fontweight="bold", color=GREEN)
ax.set_ylabel("LFW Accuracy (%)")
ax.set_ylim(85, 102)
ax.set_title("LFW Accuracy — Baseline vs AdaFace")
plt.savefig(f"{FIG_DIR}/fig6_baseline_vs_adaface.png")
plt.close()

print("Figures written to", FIG_DIR)
for f in sorted(os.listdir(FIG_DIR)):
    print("  ", f)
