import sys
from pathlib import Path

import pytest

# process-data has a hyphen so we have to add it to sys.path explicitly.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "process-data"))

from lfw_layout import find_lfw_identity_root  # noqa: E402


def _make_identity(parent: Path, name: str = "Person_A") -> None:
    person = parent / name
    person.mkdir(parents=True)
    (person / f"{name}_0001.jpg").write_bytes(b"")


def test_find_root_flat(tmp_path):
    _make_identity(tmp_path)
    assert find_lfw_identity_root(tmp_path) == tmp_path


def test_find_root_single_nested(tmp_path):
    inner = tmp_path / "lfw-deepfunneled"
    _make_identity(inner)
    assert find_lfw_identity_root(tmp_path) == inner


def test_find_root_double_nested(tmp_path):
    inner = tmp_path / "lfw-deepfunneled" / "lfw-deepfunneled"
    _make_identity(inner)
    assert find_lfw_identity_root(tmp_path) == inner


def test_find_root_funneled_variant(tmp_path):
    inner = tmp_path / "lfw_funneled"
    _make_identity(inner)
    assert find_lfw_identity_root(tmp_path) == inner


def test_find_root_no_identities_raises(tmp_path):
    (tmp_path / "some-random-file.txt").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        find_lfw_identity_root(tmp_path)
