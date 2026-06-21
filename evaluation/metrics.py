"""Canonical verification metrics from (cosine sims, labels, threshold)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(sims, labels, threshold: float) -> dict:
    sims = np.asarray(sims, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (sims > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n = len(labels)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    try:
        roc_auc = float(roc_auc_score(labels, sims))
    except ValueError:           # only one class present
        roc_auc = 0.0
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def cv_threshold_accuracy(sims, labels, n_folds: int = 10):
    """10-fold CV: tune threshold on train folds, score held-out. Returns
    (mean_acc, std_acc, median_threshold)."""
    sims = np.asarray(sims, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(sims)
    folds = n_folds if n >= n_folds else 1
    fold_size = n // folds
    cand = np.linspace(-1, 1, 401)
    accs, thresholds = [], []
    for f in range(folds):
        lo, hi = f * fold_size, (f + 1) * fold_size if f < folds - 1 else n
        val = np.zeros(n, dtype=bool)
        val[lo:hi] = True
        train = ~val if folds > 1 else val
        best_a, best_t = 0.0, 0.0
        for t in cand:
            a = ((sims[train] > t).astype(int) == labels[train]).mean()
            if a > best_a:
                best_a, best_t = a, float(t)
        accs.append(float(((sims[val] > best_t).astype(int) == labels[val]).mean()))
        thresholds.append(best_t)
    return float(np.mean(accs)), float(np.std(accs)), float(np.median(thresholds))
