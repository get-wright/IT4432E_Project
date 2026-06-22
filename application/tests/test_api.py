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


@pytest.fixture
def real_models_client(tmp_path, monkeypatch):
    """Client pointed at the real models dir; fresh data dir; threshold=-1 so any match passes."""
    real_models = Path(__file__).resolve().parents[2] / "application" / "models"
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_MODELS_DIR", str(real_models))
    monkeypatch.setenv("APP_THRESHOLD_ARCFACE", "-1.0")
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


def test_enroll_with_no_models_returns_503(client):
    r = client.post("/enroll", json={"name": "x", "image": ""})
    assert r.status_code == 503


def test_grouped_enrolled_empty_when_no_models(client):
    r = client.get("/enrolled")
    assert r.status_code == 200
    assert r.json() == []



@pytest.mark.skipif(
    not Path("application/models/arcface.pt").exists() or _first_lfw_sample() is None,
    reason="arcface checkpoint or LFW sample missing — run training first",
)
def test_enroll_all_models_and_verify_end_to_end(real_models_client):
    sample = _first_lfw_sample()
    # One capture enrolls into every available model.
    r = real_models_client.post("/enroll", json={"name": "test_user", "image": _img_b64(sample)})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "test_user"
    assert "arcface" in body["enrolled"]  # at least arcface is present

    # Per-model verify still works against arcface's own store.
    r = real_models_client.post("/verify", json={"image": _img_b64(sample), "model": "arcface"})
    assert r.status_code == 200
    assert r.json()["best_match"] == "test_user"

    # Grouped gallery: one row for the person, listing every model it landed in.
    r = real_models_client.get("/enrolled")
    person = next(e for e in r.json() if e["name"] == "test_user")
    assert set(person["models"]) == set(body["enrolled"])

    # Delete-by-name clears the person from every store.
    r = real_models_client.delete("/enrolled/by-name/test_user")
    assert r.status_code == 200 and r.json()["ok"] is True
    r = real_models_client.get("/enrolled")
    assert all(e["name"] != "test_user" for e in r.json())
