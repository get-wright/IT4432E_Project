"""Evaluate cosine-similarity verification accuracy on LFW pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .dataset import eval_transform

ROOT = Path(__file__).resolve().parents[2]


def load_pairs_txt(pairs_txt: Path) -> list[tuple[Path, Path, int]]:
    """Returns list of (img1, img2, same) where same in {0,1}.

    LFW pairs.txt format:
        first line: <n_folds> <n_per_fold>
        then for each fold:
            n_per_fold lines of "name idx1 idx2"  (positive pairs)
            n_per_fold lines of "name1 idx1 name2 idx2"  (negative pairs)
    """
    lines = pairs_txt.read_text().splitlines()
    header = lines[0].split()
    n_folds, n_per_fold = int(header[0]), int(header[1])
    img_root = ROOT / "process-data" / "lfw_pairs"
    pairs: list[tuple[Path, Path, int]] = []
    i = 1
    for _ in range(n_folds):
        for _ in range(n_per_fold):
            parts = lines[i].split(); i += 1
            name, a, b = parts[0], int(parts[1]), int(parts[2])
            p1 = img_root / name / f"{name}_{a:04d}.jpg"
            p2 = img_root / name / f"{name}_{b:04d}.jpg"
            pairs.append((p1, p2, 1))
        for _ in range(n_per_fold):
            parts = lines[i].split(); i += 1
            n1, a, n2, b = parts[0], int(parts[1]), parts[2], int(parts[3])
            p1 = img_root / n1 / f"{n1}_{a:04d}.jpg"
            p2 = img_root / n2 / f"{n2}_{b:04d}.jpg"
            pairs.append((p1, p2, 0))
    return pairs


class _PairImgDataset(Dataset):
    def __init__(self, unique_paths: list[Path]):
        self.paths = unique_paths
        self.tf = eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def evaluate_lfw(
    model,
    pairs: list[tuple[Path, Path, int]],
    device: str,
    batch_size: int = 128,
    max_pairs: int | None = None,
) -> dict:
    """Embed all referenced images, compute cosine sim per pair, run 10-fold threshold CV."""
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    unique = list({p for a, b, _ in pairs for p in (a, b)})
    path_to_idx = {p: i for i, p in enumerate(unique)}
    ds = _PairImgDataset(unique)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    model.eval()
    embs_list = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        embs_list.append(model(x).cpu())
    embs = torch.cat(embs_list, dim=0)  # already L2-normalized

    sims = []
    labels = []
    for a, b, same in pairs:
        ea = embs[path_to_idx[a]]
        eb = embs[path_to_idx[b]]
        sims.append(float((ea * eb).sum()))
        labels.append(same)
    sims_np = np.asarray(sims)
    labels_np = np.asarray(labels)

    n = len(sims_np)
    # 10-fold threshold CV. If n < 10, run a single fold.
    n_folds = 10 if n >= 10 else 1
    fold_size = n // n_folds
    cand = np.linspace(-1, 1, 401)
    accs = []
    chosen_thresholds = []
    for f in range(n_folds):
        val_lo, val_hi = f * fold_size, (f + 1) * fold_size if f < n_folds - 1 else n
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_lo:val_hi] = True
        train_mask = ~val_mask if n_folds > 1 else val_mask
        best_acc, best_t = 0.0, 0.0
        for t in cand:
            pred = (sims_np[train_mask] > t).astype(int)
            acc = (pred == labels_np[train_mask]).mean()
            if acc > best_acc:
                best_acc, best_t = acc, t
        pred = (sims_np[val_mask] > best_t).astype(int)
        accs.append(float((pred == labels_np[val_mask]).mean()))
        chosen_thresholds.append(float(best_t))

    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "threshold_global": float(np.median(chosen_thresholds)),
        "n_pairs": n,
    }
