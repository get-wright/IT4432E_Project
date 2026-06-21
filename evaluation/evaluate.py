"""Per-model LFW evaluation -> canonical metrics JSON.

Embeds pairs by loading ALREADY-ALIGNED images directly (resize to the model's
input size + per-model normalization). MTCNN is NOT re-run on aligned crops —
that belongs to the live app path only, and re-detecting inside a crop would
shift/drop faces and break comparability with the branch metrics. Missing aligned
files fall back to an 80%-center-crop of the raw image (matches the existing
ArcFace evaluator). Threshold tuned by 10-fold CV; all metrics reported at it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from application.backend.registry import REGISTRY
from .metrics import compute_metrics, cv_threshold_accuracy


def evaluate_from_sims(model: str, sims, labels) -> dict:
    # CV chooses the operating threshold; report ALL metrics (incl. accuracy) at it
    # so accuracy and tp/fp/tn/fn come from the same predictions (internally consistent).
    _, std_acc, thr = cv_threshold_accuracy(sims, labels)
    out = compute_metrics(sims, labels, thr)   # accuracy here == (tp+tn)/n at thr
    out["std_acc"] = std_acc                    # spread across folds, for reference
    out["model"] = model
    out["n_pairs"] = int(len(labels))
    return out


def _make_transform(spec):
    """Resize to the model's input size + per-model normalization. No detection."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((spec.input_size, spec.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(list(spec.mean), list(spec.std)),
    ])


def _load_aligned_or_crop(aligned: Path, raw: Path | None, tf):
    """Load aligned image directly; if missing, 80%-center-crop the raw image."""
    from PIL import Image
    if aligned.exists():
        return tf(Image.open(aligned).convert("RGB"))
    if raw is None or not raw.exists():
        raise FileNotFoundError(f"missing aligned {aligned} and no raw fallback")
    img = Image.open(raw).convert("RGB")
    w, h = img.size
    side = int(0.8 * min(w, h))
    l, t = (w - side) // 2, (h - side) // 2
    return tf(img.crop((l, t, l + side, t + side)))


def _embed_pairs(model: str, pairs, aligned_root: Path, raw_root: Path | None, device: str):
    """Embed every referenced image at the model's contract; return (sims, labels)."""
    import torch
    from application.backend.inference import Embedder

    spec = REGISTRY[model]
    ckpt = Path("application/models") / spec.ckpt_filename
    embedder = Embedder(model, ckpt, device=device)
    tf = _make_transform(spec)
    cache: dict[tuple[str, int], "torch.Tensor"] = {}

    def _emb(name: str, idx: int):
        key = (name, idx)
        if key in cache:
            return cache[key]
        fname = f"{name}_{idx:04d}.jpg"
        aligned = aligned_root / name / fname
        raw = (raw_root / name / fname) if raw_root else None
        tensor = _load_aligned_or_crop(aligned, raw, tf)
        e = embedder.embed(tensor)          # embed() normalizes + applies TTA per spec
        cache[key] = e
        return e

    sims, labels = [], []
    for p in pairs:
        e1, e2 = _emb(p.name1, p.idx1), _emb(p.name2, p.idx2)
        sims.append(float((e1 * e2).sum()))
        labels.append(p.same)
    return np.array(sims), np.array(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY))
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--aligned-root", required=True)
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from .pairs import load_pairs_txt
    pairs = load_pairs_txt(Path(args.pairs))
    sims, labels = _embed_pairs(
        args.model, pairs, Path(args.aligned_root),
        Path(args.raw_root) if args.raw_root else None, args.device,
    )
    result = evaluate_from_sims(args.model, sims, labels)
    out = Path(args.out) if args.out else Path("evaluation/results") / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


# ponytail: aligned images loaded direct + resize/normalize; MTCNN stays in the
# app path only. 80%-center-crop fallback mirrors the existing ArcFace evaluator.

if __name__ == "__main__":
    main()
