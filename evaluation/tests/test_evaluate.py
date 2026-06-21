import numpy as np
from evaluation.evaluate import evaluate_from_sims


def test_evaluate_from_sims_emits_canonical_schema():
    sims = np.concatenate([np.linspace(0.6, 0.9, 50), np.linspace(0.1, 0.4, 50)])
    labels = np.array([1] * 50 + [0] * 50)
    out = evaluate_from_sims("arcface", sims, labels)
    assert set(out) >= {"accuracy", "precision", "recall", "f1", "roc_auc",
                        "threshold", "std_acc", "tp", "fp", "tn", "fn", "model"}
    assert out["model"] == "arcface"
    assert 0.0 <= out["f1"] <= 1.0
