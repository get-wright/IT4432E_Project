"""LFW 10-fold benchmark CLI. Writes evaluation/results.json with extended metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "process-data"))

from lfw_layout import find_lfw_identity_root  # noqa: E402
from training_pipeline.src.eval_lfw import (  # noqa: E402
    evaluate_lfw,
    load_pairs_txt,
)
from training_pipeline.src.model import FaceEmbedding  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=ROOT / "application/models/best.pt")
    ap.add_argument("--pairs", default=ROOT / "preprocess-data/lfw/pairs.txt")
    ap.add_argument("--aligned-root", default=ROOT / "process-data/lfw_pairs")
    ap.add_argument("--raw-root", default=None,
                    help="Override raw LFW root; default = auto-discover under preprocess-data/lfw")
    ap.add_argument("--out", default=ROOT / "evaluation/results.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    raw_root = Path(args.raw_root) if args.raw_root else find_lfw_identity_root(ROOT / "preprocess-data/lfw")

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = FaceEmbedding(embedding_dim=ckpt["cfg"]["train"]["embedding_dim"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    pairs = load_pairs_txt(Path(args.pairs))
    metrics = evaluate_lfw(
        model, pairs,
        aligned_root=Path(args.aligned_root),
        raw_root=raw_root,
        device=args.device,
        strict=True,
    )

    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
