# evaluation/tests/test_compare.py
import json
import sys

from evaluation.compare import build_comparison, main


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


def test_main_filters_non_canonical_jsons(tmp_path, monkeypatch):
    # one canonical result + one stale pre-Task-9 file (no "model" key)
    (tmp_path / "arcface.json").write_text(json.dumps(
        {"model": "arcface", "accuracy": 0.909, "precision": 0.9, "recall": 0.92,
         "f1": 0.91, "roc_auc": 0.95, "threshold": 0.565}))
    (tmp_path / "lfw_arcface_tta.json").write_text(json.dumps(
        {"mean_acc": 0.847, "threshold_global": 0.3, "n_pairs": 6000}))

    monkeypatch.setattr(sys, "argv", ["compare", "--results-dir", str(tmp_path)])
    main()  # must not raise KeyError on the stale file

    csv = (tmp_path / "comparison.csv").read_text()
    md = (tmp_path / "comparison.md").read_text()
    assert csv.splitlines()[0] == "model,accuracy,precision,recall,f1,roc_auc,threshold"
    assert "arcface,0.909" in csv
    # stale file excluded: only header + one data row
    assert len(csv.strip().splitlines()) == 2
    assert "0.847" not in md

