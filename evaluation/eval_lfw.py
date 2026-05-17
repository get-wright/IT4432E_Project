"""Final 10-fold LFW evaluation on the best checkpoint.

Loads the best.pt checkpoint, embeds every image in LFW pairs.txt, computes
cosine similarity per pair, and runs the standard 10-fold threshold protocol.
Writes results to evaluation/results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training_pipeline.src.eval_lfw import load_pairs_txt, evaluate_lfw
from training_pipeline.src.model import FaceEmbedding

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="application/models/best.pt",
                    help="Path to trained checkpoint (relative to repo root).")
    ap.add_argument("--pairs", default="preprocess-data/lfw/lfw_funneled/pairs.txt",
                    help="Path to LFW pairs.txt (relative to repo root).")
    ap.add_argument("--out", default="evaluation/results.json")
    args = ap.parse_args()

    ckpt_path = ROOT / args.checkpoint
    pairs_path = ROOT / args.pairs
    out_path = ROOT / args.out

    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    if not pairs_path.exists():
        raise SystemExit(f"pairs.txt not found: {pairs_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = FaceEmbedding(
        embedding_dim=ckpt["cfg"]["train"]["embedding_dim"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    pairs = load_pairs_txt(pairs_path)
    metrics = evaluate_lfw(model, pairs, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
