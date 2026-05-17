"""Tests for the LFW eval loader + fallback + sanity assertions."""
from pathlib import Path

import pytest
from PIL import Image

from training_pipeline.src.eval_lfw import (
    LfwPair,
    _load_image_with_fallback,
    load_pairs_txt,
)


def _make_pairs_txt(p: Path, body: list[str]) -> None:
    p.write_text("\n".join(body) + "\n")


def _make_jpg(path: Path, size: tuple[int, int] = (250, 250)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def test_load_pairs_returns_lfwpair_objects(tmp_path):
    pairs_txt = tmp_path / "pairs.txt"
    _make_pairs_txt(pairs_txt, [
        "1 1",
        "Alice 1 2",
        "Bob 1 Carol 1",
    ])
    pairs = load_pairs_txt(pairs_txt)
    assert len(pairs) == 2
    assert all(isinstance(p, LfwPair) for p in pairs)
    assert pairs[0].name1 == "Alice" and pairs[0].name2 == "Alice" and pairs[0].same == 1
    assert pairs[1].name1 == "Bob" and pairs[1].name2 == "Carol" and pairs[1].same == 0


def test_load_pairs_does_not_drop(tmp_path):
    """Pairs are returned even when their images don't exist on disk."""
    pairs_txt = tmp_path / "pairs.txt"
    _make_pairs_txt(pairs_txt, [
        "1 2",
        "Alice 1 2",
        "Alice 3 4",
        "Bob 1 Carol 1",
        "Dave 1 Eve 1",
    ])
    pairs = load_pairs_txt(pairs_txt)
    assert len(pairs) == 4  # 2 positives + 2 negatives, none dropped


def test_load_image_with_fallback_prefers_aligned(tmp_path):
    aligned = tmp_path / "aligned" / "Alice_0001.jpg"
    raw = tmp_path / "raw" / "Alice_0001.jpg"
    _make_jpg(aligned, size=(160, 160))
    _make_jpg(raw, size=(250, 250))
    tensor, used_fallback = _load_image_with_fallback(aligned, raw)
    assert tensor.shape == (3, 160, 160)
    assert used_fallback is False


def test_load_image_with_fallback_uses_raw_when_aligned_missing(tmp_path):
    aligned = tmp_path / "aligned" / "Alice_0001.jpg"  # never created
    raw = tmp_path / "raw" / "Alice_0001.jpg"
    _make_jpg(raw, size=(250, 250))
    tensor, used_fallback = _load_image_with_fallback(aligned, raw)
    assert tensor.shape == (3, 160, 160)
    assert used_fallback is True


def test_load_image_with_fallback_raises_when_both_missing(tmp_path):
    aligned = tmp_path / "aligned" / "Alice_0001.jpg"
    raw = tmp_path / "raw" / "Alice_0001.jpg"
    with pytest.raises((AssertionError, FileNotFoundError)):
        _load_image_with_fallback(aligned, raw)
