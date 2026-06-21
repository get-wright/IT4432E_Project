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


def test_build_arcface_strict_load_rejects_wrong_keys():
    # cfg supplies embedding_dim for construction; the wrong-key model dict
    # must then fail on strict key validation.
    ckpt = {"cfg": {"train": {"embedding_dim": 512}},
            "model": {"not.a.real.key": torch.zeros(1)}}
    with pytest.raises(RuntimeError):
        REGISTRY["arcface"].builder(ckpt)


def test_build_adaface_strict_load_rejects_wrong_keys():
    # fc.weight must exist with a real shape: dim is inferred from shape[0]
    # BEFORE the load, so it has to be present (else KeyError, not RuntimeError).
    # With a 512-d fc.weight the model builds, then the extra wrong key trips
    # strict=True with a RuntimeError.
    ckpt = {"model": {"fc.weight": torch.zeros(512, 512),
                      "not.a.real.key": torch.zeros(1)}}
    with pytest.raises(RuntimeError):
        REGISTRY["adaface"].builder(ckpt)


def test_build_facenet_tolerates_vggface2_logits_head():
    # Real FaceNet checkpoints trained with classify=True carry a VGGFace2
    # classifier head (bb.logits.weight/bias) that is discarded at inference.
    # The builder must strip those keys and still strict-load the embedding
    # backbone. Regression: previously crashed with "Unexpected key(s) bb.logits.*".
    from facenet_pytorch import InceptionResnetV1
    bb = InceptionResnetV1(pretrained=None, classify=False)
    sd = {f"bb.{k}": v for k, v in bb.state_dict().items()}
    sd["bb.logits.weight"] = torch.zeros(8631, 512)  # VGGFace2 head, discarded
    sd["bb.logits.bias"] = torch.zeros(8631)
    model = REGISTRY["facenet"].builder({"model_state_dict": sd})
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 160, 160))
    assert out.shape[1] == 512
