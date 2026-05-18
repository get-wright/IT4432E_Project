"""Pins Face Recognition cross-dataset overfit check.

Pins identities are disjoint from CASIA (training) AND LFW (eval). This is the
"is the model real" test. Uses the LFW-tuned threshold as a FIXED parameter —
never re-tunes on Pins (that would just measure 'can we fit Pins?').
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training_pipeline.src.dataset import eval_transform  # noqa: E402
from training_pipeline.src.eval_lfw import _assert_distribution_sane  # noqa: E402
from training_pipeline.src.model import FaceEmbedding  # noqa: E402

N_POS = 1500
N_NEG = 1500
SEED = 0


def _build_pairs(pins_root: Path) -> list[tuple[Path, Path, int]]:
    rng = random.Random(SEED)
    celebs = sorted(d for d in pins_root.iterdir() if d.is_dir())
    by_celeb = {c.name: sorted(c.glob("*.jpg")) for c in celebs}
    by_celeb = {k: v for k, v in by_celeb.items() if len(v) >= 2}
    names = list(by_celeb.keys())

    pairs: list[tuple[Path, Path, int]] = []
    for _ in range(N_POS):
        n = rng.choice(names)
        a, b = rng.sample(by_celeb[n], 2)
        pairs.append((a, b, 1))
    for _ in range(N_NEG):
        n1, n2 = rng.sample(names, 2)
        a = rng.choice(by_celeb[n1])
        b = rng.choice(by_celeb[n2])
        pairs.append((a, b, 0))
    rng.shuffle(pairs)
    return pairs


class _PinsDataset(Dataset):
    def __init__(self, paths: list[Path], mtcnn: MTCNN):
        self.paths = paths
        self.mtcnn = mtcnn
        self.tf = eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        face = self.mtcnn(img)
        if face is None:
            # Fallback: center-crop.
            w, h = img.size
            side = int(0.8 * min(w, h))
            l = (w - side) // 2
            t = (h - side) // 2
            return self.tf(img.crop((l, t, l + side, t + side)).resize((160, 160)))
        return self.tf(Image.fromarray(face.byte().permute(1, 2, 0).cpu().numpy()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins-root",
                    default=ROOT / "preprocess-data/pins/105_classes_pins_dataset")
    ap.add_argument("--checkpoint", default=ROOT / "application/models/best.pt")
    ap.add_argument("--lfw-results", default=ROOT / "evaluation/results.json")
    ap.add_argument("--out", default=ROOT / "evaluation/results_pins.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta", action="store_true",
                    help="Use flip-averaged TTA at embed time")
    args = ap.parse_args()

    lfw = json.loads(Path(args.lfw_results).read_text())
    threshold = float(lfw["threshold_global"])
    print(f"using LFW threshold: {threshold:.4f}")

    pairs = _build_pairs(Path(args.pins_root))
    unique = sorted({p for a, b, _ in pairs for p in (a, b)})
    print(f"unique images: {len(unique)}  pairs: {len(pairs)}")

    mtcnn = MTCNN(image_size=160, margin=0, post_process=False, device=args.device, keep_all=False)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = FaceEmbedding(embedding_dim=ckpt["cfg"]["train"]["embedding_dim"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    ds = _PinsDataset(unique, mtcnn)
    dl = DataLoader(ds, batch_size=64, num_workers=0)  # MTCNN doesn't play well with workers.

    embs: list[torch.Tensor] = []
    embed_fn = model.embed_tta if args.tta else model.embed_normalized
    for x in dl:
        x = x.to(args.device, non_blocking=True)
        with torch.no_grad():
            embs.append(embed_fn(x).cpu())
    embs_t = torch.cat(embs, dim=0)
    path_to_idx = {p: i for i, p in enumerate(unique)}

    sims = np.array([float((embs_t[path_to_idx[a]] * embs_t[path_to_idx[b]]).sum())
                     for a, b, _ in pairs])
    labels = np.array([l for _, _, l in pairs])

    pos = sims[labels == 1]
    neg = sims[labels == 0]
    metrics = {
        "dataset": "pins-face-recognition (105 celebs, disjoint from CASIA + LFW)",
        "n_pairs": int(len(sims)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "pos_sim_mean": float(pos.mean()),
        "pos_sim_std": float(pos.std()),
        "neg_sim_mean": float(neg.mean()),
        "neg_sim_std": float(neg.std()),
        "spread": float(pos.mean() - neg.mean()),
        "pos_ratio": float(labels.mean()),
        "lfw_threshold_used": threshold,
        "accuracy_at_lfw_threshold": float(((sims > threshold).astype(int) == labels).mean()),
    }
    # Reference-only: best-on-pins threshold (NOT the headline number).
    cand = np.linspace(-1, 1, 401)
    accs = np.array([((sims > t).astype(int) == labels).mean() for t in cand])
    metrics["accuracy_at_pins_tuned_threshold"] = float(accs.max())
    metrics["pins_tuned_threshold"] = float(cand[int(accs.argmax())])

    _assert_distribution_sane(metrics)  # distribution-only; no threshold-bound check.

    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
