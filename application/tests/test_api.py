"""End-to-end FastAPI integration test. Skipped when checkpoint missing."""
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_THRESHOLD", "-1.0")  # so any match passes
    from application.backend import main as app_main
    app_main._reset_for_tests()
    return TestClient(app_main.app)


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _first_lfw_sample() -> Path | None:
    root = Path("process-data/lfw_pairs")
    if not root.exists():
        return None
    for p in root.rglob("*.jpg"):
        return p
    return None


@pytest.mark.skipif(
    not Path("application/models/best.pt").exists() or _first_lfw_sample() is None,
    reason="checkpoint or LFW sample missing — run training first",
)
def test_enroll_and_verify_end_to_end(client):
    sample = _first_lfw_sample()
    r = client.post("/enroll", json={"name": "test_user", "image": _img_b64(sample)})
    assert r.status_code == 200
    enrolled_id = r.json()["id"]

    r = client.post("/verify", json={"image": _img_b64(sample)})
    assert r.status_code == 200
    assert r.json()["best_match"] == "test_user"

    r = client.get("/enrolled")
    assert any(e["name"] == "test_user" for e in r.json())

    r = client.delete(f"/enrolled/{enrolled_id}")
    assert r.status_code == 200
    r = client.get("/enrolled")
    assert all(e["name"] != "test_user" for e in r.json())
