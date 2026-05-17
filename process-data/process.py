"""MTCNN-align all faces, build train/val manifest, prepare LFW pairs."""
from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "preprocess-data"
OUT = ROOT / "process-data"
IMG_EXT = {".jpg", ".jpeg", ".png"}

log = logging.getLogger("process")


def _find_casia_root() -> tuple[Path, str]:
    """Locate the directory containing CASIA images.

    Returns (dir, layout) where layout is one of:
        "nested" — `<dir>/<identity_id>/<file>.jpg`
        "flat"   — `<dir>/<identity_id>_<num>.jpg`
    """
    base = RAW / "casia-webface"
    candidates = [base, *(p for p in base.iterdir() if p.is_dir())]
    for c in candidates:
        try:
            entries = list(c.iterdir())
        except Exception:
            continue
        # Nested: many subdirs of images.
        subdirs = [d for d in entries if d.is_dir()]
        if len(subdirs) > 100:
            return c, "nested"
        # Flat: many image files with id-prefixed names.
        files = [f for f in entries if f.is_file() and f.suffix.lower() in IMG_EXT]
        if len(files) > 1000 and "_" in files[0].stem:
            return c, "flat"
    raise SystemExit("CASIA-WebFace identity root not found")


def build_pairs_casia() -> list[tuple[Path, str]]:
    casia_root, layout = _find_casia_root()
    pairs: list[tuple[Path, str]] = []
    if layout == "nested":
        for ident_dir in casia_root.iterdir():
            if not ident_dir.is_dir():
                continue
            for img in ident_dir.iterdir():
                if img.suffix.lower() in IMG_EXT:
                    pairs.append((img, f"casia_{ident_dir.name}"))
    else:  # flat: identity_<num>.jpg
        for img in casia_root.iterdir():
            if img.is_file() and img.suffix.lower() in IMG_EXT and "_" in img.stem:
                ident = img.stem.rsplit("_", 1)[0]
                pairs.append((img, f"casia_{ident}"))
    return pairs


def build_pairs_celeba() -> list[tuple[Path, str]]:
    identity_file = next((RAW / "celeba").rglob("identity_CelebA.txt"))
    img_dir = next((RAW / "celeba").rglob("img_align_celeba"))
    img_to_id: dict[str, str] = {}
    with identity_file.open() as f:
        for line in f:
            img, ident = line.strip().split()
            img_to_id[img] = f"celeba_{ident}"
    pairs = []
    for img in img_dir.iterdir():
        if img.name in img_to_id:
            pairs.append((img, img_to_id[img.name]))
    return pairs


def build_pairs_lfw() -> list[tuple[Path, str]]:
    base = RAW / "lfw"
    deep = next(base.rglob("lfw-deepfunneled"), None) or next(base.rglob("lfw_funneled"), None)
    if deep is None:
        raise SystemExit("LFW image root not found")
    pairs = []
    for ident_dir in deep.iterdir():
        if ident_dir.is_dir():
            for img in ident_dir.iterdir():
                if img.suffix.lower() in IMG_EXT:
                    pairs.append((img, f"lfw_{ident_dir.name}"))
    return pairs


def align_batch(mtcnn: MTCNN, src_paths: list[Path], out_paths: list[Path]) -> list[bool]:
    """Open images, run MTCNN one-by-one, save aligned crops.

    facenet-pytorch 2.6.0 has a NumPy >=1.24 incompatibility for batched calls
    (inhomogeneous array). Single-image calls work fine and on H100 still hit
    ~50-100 imgs/sec.
    """
    ok: list[bool] = []
    for sp, op in zip(src_paths, out_paths):
        try:
            img = Image.open(sp).convert("RGB")
        except Exception:
            ok.append(False)
            continue
        try:
            aligned = mtcnn(img, save_path=str(op))
        except Exception:
            aligned = None
        ok.append(aligned is not None)
    return ok


def process_split(
    name: str,
    pairs: list[tuple[Path, str]],
    split_for_train: bool,
    mtcnn: MTCNN,
    args: argparse.Namespace,
    train_root: Path,
    val_root: Path,
    lfw_root: Path,
    failures_writer,
    manifest_rows: list[dict],
) -> None:
    by_id: dict[str, list[Path]] = {}
    for p, ident in pairs:
        by_id.setdefault(ident, []).append(p)
    for ident in list(by_id):
        random.shuffle(by_id[ident])
        by_id[ident] = by_id[ident][: args.limit_per_id]
        if len(by_id[ident]) < args.min_per_id:
            del by_id[ident]

    for ident, paths in tqdm(by_id.items(), desc=f"align {name}"):
        n = len(paths)
        n_val = max(1, int(n * args.val_frac)) if split_for_train else 0
        for idx in range(0, n, args.batch):
            batch_paths = paths[idx: idx + args.batch]
            out_paths = []
            for j, sp in enumerate(batch_paths):
                global_idx = idx + j
                if split_for_train and global_idx < n_val:
                    d = val_root / ident
                elif split_for_train:
                    d = train_root / ident
                else:
                    d = lfw_root / ident.removeprefix("lfw_")
                d.mkdir(parents=True, exist_ok=True)
                out_paths.append(d / sp.name)
            ok = align_batch(mtcnn, batch_paths, out_paths)
            for j, (sp, op, ok_flag) in enumerate(zip(batch_paths, out_paths, ok)):
                global_idx = idx + j
                split = ("val" if split_for_train and global_idx < n_val else
                         "train" if split_for_train else "lfw")
                if ok_flag:
                    manifest_rows.append({
                        "path": str(op.relative_to(ROOT)),
                        "identity_id": ident,
                        "source": name,
                        "split": split,
                    })
                else:
                    failures_writer.writerow([str(sp.relative_to(ROOT)), "mtcnn_no_face"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-per-id", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit-per-id", type=int, default=80,
                    help="Cap images per identity to control training set size.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-casia", action="store_true")
    ap.add_argument("--skip-celeba", action="store_true")
    ap.add_argument("--skip-lfw", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device %s", device)

    mtcnn = MTCNN(image_size=160, margin=0, post_process=False, device=device, keep_all=False)

    train_root = OUT / "train"
    val_root = OUT / "val"
    lfw_root = OUT / "lfw_pairs"
    for d in (train_root, val_root, lfw_root):
        d.mkdir(parents=True, exist_ok=True)

    failures_path = OUT / "alignment_failures.csv"
    with open(failures_path, "w", newline="") as failures_file:
        fw = csv.writer(failures_file)
        fw.writerow(["source_path", "reason"])

        manifest_rows: list[dict] = []

        if not args.skip_casia:
            log.info("Building CASIA pairs...")
            process_split("casia", build_pairs_casia(), True, mtcnn, args,
                          train_root, val_root, lfw_root, fw, manifest_rows)
        if not args.skip_celeba:
            log.info("Building CelebA pairs...")
            process_split("celeba", build_pairs_celeba(), True, mtcnn, args,
                          train_root, val_root, lfw_root, fw, manifest_rows)
        if not args.skip_lfw:
            log.info("Building LFW pairs...")
            process_split("lfw", build_pairs_lfw(), False, mtcnn, args,
                          train_root, val_root, lfw_root, fw, manifest_rows)

    df = pd.DataFrame(manifest_rows)
    df.to_parquet(OUT / "manifest.parquet", index=False)
    log.info("Wrote %d rows to manifest.parquet", len(df))
    log.info("Train: %d  Val: %d  LFW: %d",
             int((df["split"] == "train").sum()),
             int((df["split"] == "val").sum()),
             int((df["split"] == "lfw").sum()))


if __name__ == "__main__":
    main()
