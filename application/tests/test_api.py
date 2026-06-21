"""FastAPI integration tests. No-weight tests run always; e2e skipped without checkpoint."""
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_MODELS_DIR", str(tmp_path))  # no checkpoints → all unavailable
    from application.backend import main as app_main
    app_main._reset_for_tests()
    return TestClient(app_main.app)


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _first_lfw_sample() -> Path | None:
    root = Path("shared/process-data/lfw_pairs")
    if not root.exists():
        return None
    for p in root.rglob("*.jpg"):
        return p
    return None


def test_models_endpoint_lists_all_registry_entries(client):
    r = client.get("/models")
    assert r.status_code == 200
    entries = r.json()
    names = {m["name"] for m in entries}
    assert names == {"arcface", "adaface", "facenet"}
    # No checkpoints present in tmp env → all unavailable, none crash.
    assert all(m["available"] is False for m in entries)
    # Full contract: every entry carries threshold + is_default; unavailable → reason set.
    for m in entries:
        assert "threshold" in m
        assert "is_default" in m
        assert m["reason"] == "checkpoint missing"


def test_verify_unavailable_model_returns_503(client):
    r = client.post("/verify", json={"image": "", "model": "arcface"})
    assert r.status_code == 503


@pytest.mark.skipif(
    not Path("application/models/arcface.pt").exists() or _first_lfw_sample() is None,
    reason="arcface checkpoint or LFW sample missing — run training first",
)
def test_enroll_and_verify_end_to_end(client):
    sample = _first_lfw_sample()
    r = client.post("/enroll", json={"name": "test_user", "image": _img_b64(sample), "model": "arcface"})
    assert r.status_code == 200
    enrolled_id = r.json()["id"]

    r = client.post("/verify", json={"image": _img_b64(sample), "model": "arcface"})
    assert r.status_code == 200
    assert r.json()["best_match"] == "test_user"

    r = client.get("/enrolled", params={"model": "arcface"})
    assert any(e["name"] == "test_user" for e in r.json())

    r = client.delete(f"/enrolled/{enrolled_id}", params={"model": "arcface"})
    assert r.status_code == 200
    r = client.get("/enrolled", params={"model": "arcface"})
    assert all(e["name"] != "test_user" for e in r.json())
