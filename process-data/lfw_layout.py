"""Discover the directory containing LFW identity folders.

The Kaggle LFW dumps ship with inconsistent nesting:
- flat:                <root>/<Person>/<Person>_0001.jpg
- single-nested:       <root>/lfw-deepfunneled/<Person>/...
- double-nested:       <root>/lfw-deepfunneled/lfw-deepfunneled/<Person>/...
- funneled variant:    <root>/lfw_funneled/<Person>/...

`find_lfw_identity_root` walks down from `raw_dir` until it finds a directory
whose children look like identity folders (a subdir containing `<name>_NNNN.jpg`).
"""
from __future__ import annotations

from pathlib import Path


def _looks_like_identity_dir(d: Path) -> bool:
    """A directory whose name matches its file prefix, e.g. 'Aaron_Eckhart' contains 'Aaron_Eckhart_0001.jpg'."""
    if not d.is_dir():
        return False
    for f in d.iterdir():
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if f.name.startswith(d.name + "_"):
                return True
    return False


def find_lfw_identity_root(raw_dir: Path) -> Path:
    """Find the directory that directly contains LFW identity folders.

    Walks down at most 3 levels of nesting. Raises FileNotFoundError if no
    identity-shaped directory is found.
    """
    raw_dir = Path(raw_dir)
    candidates = [raw_dir]
    for depth in range(3):
        next_candidates: list[Path] = []
        for c in candidates:
            if not c.is_dir():
                continue
            for sub in sorted(c.iterdir()):
                if _looks_like_identity_dir(sub):
                    return c
            next_candidates.extend(s for s in c.iterdir() if s.is_dir())
        candidates = next_candidates
        if not candidates:
            break
    raise FileNotFoundError(f"No LFW identity folders found under {raw_dir}")
