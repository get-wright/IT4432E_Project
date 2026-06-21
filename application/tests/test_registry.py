# application/tests/test_registry.py
import pytest
import torch
from application.backend.registry import REGISTRY, ModelSpec


def test_registry_has_three_models():
    assert set(REGISTRY) == {"arcface", "adaface", "facenet"}


def test_per_model_contracts_match_spec():
    arc, ada, fac = REGISTRY["arcface"], REGISTRY["adaface"], REGISTRY["facenet"]
    assert arc.input_size == 160 and arc.mean == (0.485, 0.456, 0.406)
    assert arc.threshold == 0.565 and arc.use_tta is False and arc.needs_cfg is True
    assert ada.input_size == 112 and ada.mean == (0.5, 0.5, 0.5) and ada.ckpt_key == "model"
    assert fac.input_size == 160 and fac.mean == (0.5, 0.5, 0.5)
    assert fac.ckpt_key == "model_state_dict"


def test_build_facenet_strict_load_rejects_wrong_keys():
    # Tensor value under a wrong key → must fail on strict key validation,
    # not earlier on a non-tensor value.
    with pytest.raises(RuntimeError):
        REGISTRY["facenet"].builder({"model_state_dict": {"not.a.real.key": torch.zeros(1)}})
