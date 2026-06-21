"""Runs the 8-suite InsightFace .bin verification benchmarks (lfw, agedb_30,
cfp_ff, cfp_fp, cplfw, calfw, sllfw, talfw) against a checkpoint and writes a JSON
with one entry per benchmark. Schema matches results_insightface_bench.json so
benchmarks.ipynb keeps reading both old and new results without changes.
"""
from __future__ import annotations

import argparse
import json
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.arcface.dataset import eval_transform
from models.arcface.model import FaceEmbedding

BENCH_NAMES = ["lfw", "agedb_30", "cfp_ff", "cfp_fp", "cplfw", "calfw", "sllfw", "talfw"]
LFW_TUNED_THRESHOLD = 0.565


def _load_bin(bin_path: Path) -> tuple[list[Image.Image], np.ndarray]:
    """InsightFace .bin layout: pickle of (image_byte_list, issame_list).

    image_byte_list is a flat list of length 2*N: pairs are (img[2i], img[2i+1]).
    issame_list has length N.
    """
    with open(bin_path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")
    images = [Image.open(BytesIO(b)).convert("RGB") for b in bins]
    return images, np.array(issame, dtype=bool)


@torch.no_grad()
def _embed_all(model: FaceEmbedding, images: list[Image.Image], device: str,
               use_tta: bool, batch_size: int = 128) -> torch.Tensor:
    # Reuse the exact eval transform the training pipeline uses (ImageNet mean/std,
    # 160x160 resize, ToTensor). This is what produced existing eval numbers.
    tf = eval_transform()
    embed_fn = model.embed_tta if use_tta else model.embed_normalized
    out = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([tf(im) for im in images[i:i + batch_size]]).to(device)
        out.append(embed_fn(batch).cpu())
    return torch.cat(out, dim=0)


def _ten_fold_cv_accuracy(sims: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """LFW-style 10-fold CV: pick best threshold on each train fold, evaluate on val fold.

    Returns (mean_acc, std_acc, mean_chosen_threshold).
    """
    n = len(sims)
    fold_size = n // 10
    cand = np.linspace(-1.0, 1.0, 401)
    accs, thrs = [], []
    for k in range(10):
        lo, hi = k * fold_size, (k + 1) * fold_size if k < 9 else n
        val_idx = np.zeros(n, dtype=bool)
        val_idx[lo:hi] = True
        train_sims, train_lab = sims[~val_idx], labels[~val_idx]
        train_accs = ((train_sims[None, :] >= cand[:, None]) == train_lab[None, :]).mean(axis=1)
        best = float(cand[int(train_accs.argmax())])
        thrs.append(best)
        val_acc = ((sims[val_idx] >= best) == labels[val_idx]).mean()
        accs.append(float(val_acc))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(thrs))


def _run_one(model: FaceEmbedding, bin_path: Path, device: str, use_tta: bool) -> dict:
    images, issame = _load_bin(bin_path)
    embs = _embed_all(model, images, device, use_tta)
    # Pair (2i, 2i+1)
    e_a = embs[0::2]
    e_b = embs[1::2]
    sims = (e_a * e_b).sum(dim=1).numpy()
    labels = issame.astype(bool)
    pos = sims[labels]
    neg = sims[~labels]
    mean_acc, std_acc, _ = _ten_fold_cv_accuracy(sims, labels)
    acc_at_lfw_thr = float(((sims >= LFW_TUNED_THRESHOLD) == labels).mean())
    # Schema matches evaluation/results_insightface_bench.json (n_pairs, n_pos, n_neg, mean_acc_cv, …).
    return {
        "n_pairs": int(len(sims)),
        "n_pos": int(labels.sum()),
        "n_neg": int((~labels).sum()),
        "mean_acc_cv": mean_acc,
        "std_acc_cv": std_acc,
        f"acc_at_lfw_threshold_{LFW_TUNED_THRESHOLD}": acc_at_lfw_thr,
        "pos_sim_mean": float(pos.mean()),
        "neg_sim_mean": float(neg.mean()),
        "spread": float(pos.mean() - neg.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--bins-root", type=Path, default=Path("preshared/process-data/insightface_bins"),
                    help="Directory containing <name>.bin files for each benchmark")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bench", nargs="*", default=BENCH_NAMES,
                    help=f"Subset of benchmarks to run. Default: {BENCH_NAMES}")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = FaceEmbedding(embedding_dim=ckpt["cfg"]["train"]["embedding_dim"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    results = {}
    for name in args.bench:
        bin_path = args.bins_root / f"{name}.bin"
        if not bin_path.exists():
            print(f"SKIP {name}: {bin_path} not found")
            continue
        print(f"running {name}…")
        results[name] = _run_one(model, bin_path, args.device, args.tta)
        print(f"  acc_cv={results[name]['mean_acc_cv']:.4f}  "
              f"spread={results[name]['spread']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
