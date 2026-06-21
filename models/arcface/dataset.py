from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FaceDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, transform=None):
        df = pd.read_parquet(manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)
        ids = sorted(df["identity_id"].unique())
        self.id_to_idx = {i: n for n, i in enumerate(ids)}
        df["label"] = df["identity_id"].map(self.id_to_idx)
        self.df = df
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(ROOT / row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


class PKSampler(Sampler):
    """Yield batches of P identities x K samples each. Identities with < K samples are skipped."""

    def __init__(self, labels: Sequence[int], p: int, k: int, num_batches: int, seed: int = 0):
        self.labels = list(labels)
        self.p = p
        self.k = k
        self.num_batches = num_batches
        self.seed = seed
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.by_label[lbl].append(idx)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        label_pool = [lbl for lbl, idxs in self.by_label.items() if len(idxs) >= self.k]
        for _ in range(self.num_batches):
            chosen = rng.sample(label_pool, self.p)
            batch: list[int] = []
            for lbl in chosen:
                batch.extend(rng.sample(self.by_label[lbl], self.k))
            yield batch

    def __len__(self) -> int:
        return self.num_batches
