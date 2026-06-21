import numpy as np
from evaluation.metrics import compute_metrics, cv_threshold_accuracy


def test_perfect_separation():
    m = compute_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 0, 2, 0)
    assert m["accuracy"] == 1.0 and m["precision"] == 1.0
    assert m["recall"] == 1.0 and m["f1"] == 1.0


def test_mixed_case_hand_computed():
    # preds at thr .5: [1,0,1,0]; labels [1,1,0,0]
    # idx0 tp, idx1 fn, idx2 fp, idx3 tn  -> acc .5 p .5 r .5 f1 .5
    m = compute_metrics([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0], threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5
    assert m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["threshold"] == 0.5


def test_cv_threshold_recovers_separating_threshold():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.6, 0.9, 200)
    neg = rng.uniform(0.1, 0.4, 200)
    sims = np.concatenate([pos, neg])
    labels = np.array([1] * 200 + [0] * 200)
    mean_acc, std_acc, thr = cv_threshold_accuracy(sims, labels)
    assert mean_acc > 0.95 and 0.4 <= thr <= 0.6
