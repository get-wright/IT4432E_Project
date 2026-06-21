# evaluation/tests/test_compare.py
from evaluation.compare import build_comparison


def test_comparison_table_has_all_models_and_metrics():
    results = [
        {"model": "arcface", "accuracy": 0.909, "precision": 0.9, "recall": 0.92,
         "f1": 0.91, "roc_auc": 0.95, "threshold": 0.565},
        {"model": "adaface", "accuracy": 0.994, "precision": 0.996, "recall": 0.991,
         "f1": 0.994, "roc_auc": 0.999, "threshold": 0.237},
    ]
    md, csv = build_comparison(results)
    assert "arcface" in md and "adaface" in md
    assert "| model |" in md.lower() or "model" in md.splitlines()[0]
    assert csv.splitlines()[0] == "model,accuracy,precision,recall,f1,roc_auc,threshold"
    assert "arcface,0.909" in csv
