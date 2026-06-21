"""Tests for the LFW eval loader + fallback + sanity assertions."""
from pathlib import Path

import pytest
from PIL import Image

from models.arcface.eval_lfw import (
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


from models.arcface.eval_lfw import (
    _assert_distribution_sane,
    _assert_threshold_sane,
)


def _metrics(pos_mean=0.7, neg_mean=0.3, pos_std=0.05, pos_ratio=0.5, threshold=0.5):
    return {
        "pos_sim_mean": pos_mean,
        "neg_sim_mean": neg_mean,
        "spread": pos_mean - neg_mean,
        "pos_sim_std": pos_std,
        "neg_sim_std": 0.05,
        "pos_ratio": pos_ratio,
        "threshold_global": threshold,
    }


def test_distribution_sane_passes_on_real():
    _assert_distribution_sane(_metrics())  # no exception


def test_distribution_collapse_fires_on_low_spread():
    with pytest.raises(AssertionError, match="COLLAPSED"):
        _assert_distribution_sane(_metrics(pos_mean=0.997, neg_mean=0.996))


def test_distribution_collapse_fires_on_tight_std():
    with pytest.raises(AssertionError, match="too tight"):
        _assert_distribution_sane(_metrics(pos_std=0.001))


def test_distribution_label_leak_fires():
    with pytest.raises(AssertionError, match="LABEL LEAK"):
        _assert_distribution_sane(_metrics(pos_ratio=0.97))


def test_threshold_sane_passes_on_real():
    _assert_threshold_sane(_metrics(threshold=0.45))


def test_threshold_at_lower_bound_fires():
    with pytest.raises(AssertionError, match="THRESHOLD AT BOUND"):
        _assert_threshold_sane(_metrics(threshold=-1.0))


def test_threshold_at_upper_bound_fires():
    with pytest.raises(AssertionError, match="THRESHOLD AT BOUND"):
        _assert_threshold_sane(_metrics(threshold=0.95))


def test_evaluate_lfw_use_tta_calls_embed_tta(monkeypatch, tmp_path):
    """use_tta=True must dispatch to model.embed_tta instead of embed_normalized."""
    import torch
    from models.arcface.eval_lfw import evaluate_lfw, LfwPair

    calls = {"embed_normalized": 0, "embed_tta": 0}

    class _SpyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 160 * 160, 4)
        def embed_normalized(self, x):
            calls["embed_normalized"] += 1
            return torch.nn.functional.normalize(self.fc(x.flatten(1)), dim=1)
        def embed_tta(self, x):
            calls["embed_tta"] += 1
            return torch.nn.functional.normalize(self.fc(x.flatten(1)), dim=1)

    # Stub pair list with the dataset wrapper missing → use raw fallback for everything.
    # Easiest: monkeypatch loader to return a deterministic tensor per pair.
    # See existing test pattern in this file for how to build a 2-pair fixture.
    pairs = [
        LfwPair(name1="A", idx1=1, name2="A", idx2=2, same=True),
        LfwPair(name1="A", idx1=1, name2="B", idx2=1, same=False),
    ]

    class _StubDataset:
        def __init__(self, *_a, **_k): pass
        def __len__(self): return 3
        def __getitem__(self, i): return torch.zeros(3, 160, 160)

    from models.arcface import eval_lfw as ev
    monkeypatch.setattr(ev, "_PairImgDataset", _StubDataset)

    model = _SpyModel().eval()
    evaluate_lfw(
        model, pairs,
        aligned_root=tmp_path, raw_root=tmp_path,
        device="cpu", strict=False, use_tta=True,
    )
    assert calls["embed_tta"] >= 1
    assert calls["embed_normalized"] == 0
