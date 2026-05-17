"""LFW verification: load pairs, embed via fallback-capable image loader,
compute cosine similarity, run 10-fold threshold CV, optionally enforce sanity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .dataset import IMAGENET_MEAN, IMAGENET_STD, eval_transform

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LfwPair:
    name1: str
    idx1: int
    name2: str
    idx2: int
    same: int  # 0 or 1


def load_pairs_txt(pairs_txt: Path) -> list[LfwPair]:
    """Parse LFW pairs.txt and return ALL pairs as LfwPair objects.

    Does NOT filter by disk presence — that's the eval's job, with fallback to raw.
    Format:
        <n_folds> <n_per_fold>
        n_per_fold lines of "name idx1 idx2"      (positives)
        n_per_fold lines of "name1 idx1 name2 idx2" (negatives)
        ... repeated for each fold
    """
    lines = pairs_txt.read_text().splitlines()
    header = lines[0].split()
    n_folds, n_per_fold = int(header[0]), int(header[1])
    pairs: list[LfwPair] = []
    i = 1
    for _ in range(n_folds):
        for _ in range(n_per_fold):
            parts = lines[i].split(); i += 1
            pairs.append(LfwPair(parts[0], int(parts[1]), parts[0], int(parts[2]), 1))
        for _ in range(n_per_fold):
            parts = lines[i].split(); i += 1
            pairs.append(LfwPair(parts[0], int(parts[1]), parts[2], int(parts[3]), 0))
    return pairs


_raw_fallback_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _load_image_with_fallback(aligned: Path, raw: Path) -> tuple[torch.Tensor, bool]:
    """Load aligned if it exists, otherwise center-crop raw to 80% of min(w,h) and resize.

    Returns (tensor, used_fallback). Missing raw is a setup bug; raises AssertionError.
    """
    if aligned.exists():
        return eval_transform()(Image.open(aligned).convert("RGB")), False
    assert raw.exists(), f"raw fallback also missing: {raw}"
    img = Image.open(raw).convert("RGB")
    w, h = img.size
    side = int(0.8 * min(w, h))
    l = (w - side) // 2
    t = (h - side) // 2
    img = img.crop((l, t, l + side, t + side)).resize((160, 160), Image.BILINEAR)
    return _raw_fallback_tf(img), True


def _pair_paths(pair: LfwPair, aligned_root: Path, raw_root: Path) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    a1 = aligned_root / pair.name1 / f"{pair.name1}_{pair.idx1:04d}.jpg"
    r1 = raw_root / pair.name1 / f"{pair.name1}_{pair.idx1:04d}.jpg"
    a2 = aligned_root / pair.name2 / f"{pair.name2}_{pair.idx2:04d}.jpg"
    r2 = raw_root / pair.name2 / f"{pair.name2}_{pair.idx2:04d}.jpg"
    return (a1, r1), (a2, r2)


class _PairImgDataset(Dataset):
    """Loads each unique image; tracks aligned vs raw-fallback counts via a shared list."""

    def __init__(self, unique: list[tuple[Path, Path]], fallback_log: list[bool]):
        self.unique = unique
        self.fallback_log = fallback_log

    def __len__(self) -> int:
        return len(self.unique)

    def __getitem__(self, i: int):
        aligned, raw = self.unique[i]
        tensor, used_fallback = _load_image_with_fallback(aligned, raw)
        self.fallback_log.append(used_fallback)
        return tensor


def _assert_distribution_sane(metrics: dict) -> None:
    assert metrics["spread"] > 0.05, (
        f"COLLAPSED: spread={metrics['spread']:.4f} "
        f"(pos={metrics['pos_sim_mean']:.3f}, neg={metrics['neg_sim_mean']:.3f})"
    )
    assert metrics["pos_sim_std"] > 0.01, (
        f"COLLAPSED: pos std={metrics['pos_sim_std']:.4f} too tight"
    )
    assert 0.4 < metrics["pos_ratio"] < 0.6, (
        f"LABEL LEAK: {metrics['pos_ratio']:.2%} positive"
    )


def _assert_threshold_sane(metrics: dict) -> None:
    assert 0.0 < metrics["threshold_global"] < 0.9, (
        f"THRESHOLD AT BOUND: {metrics['threshold_global']:.4f}"
    )


@torch.no_grad()
def evaluate_lfw(
    model,
    pairs: list[LfwPair],
    aligned_root: Path,
    raw_root: Path,
    device: str,
    batch_size: int = 128,
    max_pairs: int | None = None,
    strict: bool = True,
) -> dict:
    """Embed all referenced images (with raw fallback), compute cosine sim per pair,
    run 10-fold threshold CV. If `strict`, enforce distribution + threshold sanity.

    The model must implement `embed_normalized(x)` returning unit-norm vectors.
    """
    n_pairs_total = len(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    by_key: dict[tuple[str, int], int] = {}
    unique: list[tuple[Path, Path]] = []

    def _key(name: str, idx: int) -> tuple[str, int]:
        return (name, idx)

    for pair in pairs:
        (a1, r1), (a2, r2) = _pair_paths(pair, aligned_root, raw_root)
        if _key(pair.name1, pair.idx1) not in by_key:
            by_key[_key(pair.name1, pair.idx1)] = len(unique)
            unique.append((a1, r1))
        if _key(pair.name2, pair.idx2) not in by_key:
            by_key[_key(pair.name2, pair.idx2)] = len(unique)
            unique.append((a2, r2))

    fallback_log: list[bool] = []
    ds = _PairImgDataset(unique, fallback_log)
    # num_workers=0 so the shared fallback_log isn't duplicated per worker.
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)

    model.eval()
    embs_list: list[torch.Tensor] = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        embs_list.append(model.embed_normalized(x).cpu())
    embs = torch.cat(embs_list, dim=0)

    sims = np.array([
        float((embs[by_key[_key(p.name1, p.idx1)]] * embs[by_key[_key(p.name2, p.idx2)]]).sum())
        for p in pairs
    ])
    labels = np.array([p.same for p in pairs])

    n = len(sims)
    n_folds = 10 if n >= 10 else 1
    fold_size = n // n_folds
    cand = np.linspace(-1, 1, 401)
    accs: list[float] = []
    chosen_thresholds: list[float] = []
    for f in range(n_folds):
        lo, hi = f * fold_size, (f + 1) * fold_size if f < n_folds - 1 else n
        val = np.zeros(n, dtype=bool); val[lo:hi] = True
        train = ~val if n_folds > 1 else val
        best_a, best_t = 0.0, 0.0
        for t in cand:
            a = ((sims[train] > t).astype(int) == labels[train]).mean()
            if a > best_a:
                best_a, best_t = a, float(t)
        accs.append(float(((sims[val] > best_t).astype(int) == labels[val]).mean()))
        chosen_thresholds.append(best_t)

    pos = sims[labels == 1]
    neg = sims[labels == 0]
    n_raw_fallback = sum(fallback_log)
    metrics = {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "threshold_global": float(np.median(chosen_thresholds)),
        "n_pairs": n,
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_used": int(n),
        "n_unique_images": int(len(unique)),
        "n_aligned": int(len(unique) - n_raw_fallback),
        "n_raw_fallback": int(n_raw_fallback),
        "pos_sim_mean": float(pos.mean()) if len(pos) else 0.0,
        "pos_sim_std": float(pos.std()) if len(pos) else 0.0,
        "neg_sim_mean": float(neg.mean()) if len(neg) else 0.0,
        "neg_sim_std": float(neg.std()) if len(neg) else 0.0,
        "spread": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else 0.0,
        "pos_ratio": float(labels.mean()),
    }

    if strict:
        _assert_distribution_sane(metrics)
        _assert_threshold_sane(metrics)

    return metrics
