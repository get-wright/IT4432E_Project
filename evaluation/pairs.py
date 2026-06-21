"""LFW pairs.txt loader (lifted from the ArcFace eval path)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LfwPair:
    name1: str
    idx1: int
    name2: str
    idx2: int
    same: int


def load_pairs_txt(pairs_txt: Path) -> list[LfwPair]:
    lines = Path(pairs_txt).read_text().strip().splitlines()
    pairs: list[LfwPair] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) == 3:        # same person: name idx1 idx2
            pairs.append(LfwPair(parts[0], int(parts[1]), parts[0], int(parts[2]), 1))
        elif len(parts) == 4:      # different: name1 idx1 name2 idx2
            pairs.append(LfwPair(parts[0], int(parts[1]), parts[2], int(parts[3]), 0))
    return pairs
