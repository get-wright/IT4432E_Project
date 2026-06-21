"""Aggregate per-model result JSONs into one comparison table (md + csv)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]


def build_comparison(results: list[dict]) -> tuple[str, str]:
    rows = sorted(results, key=lambda r: r["model"])

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)

    header = "| " + " | ".join(COLS) + " |"
    sep = "|" + "|".join(["---"] * len(COLS)) + "|"
    md_lines = [header, sep]
    csv_lines = [",".join(COLS)]
    for r in rows:
        cells = [r["model"]] + [fmt(r.get(c, "")) for c in COLS[1:]]
        md_lines.append("| " + " | ".join(cells) + " |")
        csv_lines.append(",".join(cells))
    return "\n".join(md_lines) + "\n", "\n".join(csv_lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="evaluation/results")
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    raw = [json.loads(p.read_text()) for p in rdir.glob("*.json")
           if p.name not in {"comparison.json"}]
    results = [d for d in raw if "model" in d]
    md, csv = build_comparison(results)
    (rdir / "comparison.md").write_text(md)
    (rdir / "comparison.csv").write_text(csv)
    print(md)


if __name__ == "__main__":
    main()
