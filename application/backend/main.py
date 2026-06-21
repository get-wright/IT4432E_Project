"""FastAPI app: multi-model enroll / verify / list / delete + static frontend."""
from __future__ import annotations

import base64
import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import EnrollmentDB
from .face_align import FaceAligner
from .inference import Embedder
from .registry import REGISTRY

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "application" / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _reset_for_tests()
    yield


app = FastAPI(title="Face Recognition", lifespan=lifespan)

# name -> {"embedder", "aligner", "db", "threshold"} for available models only
_loaded: dict[str, dict] = {}
_default_model: str | None = None


def _config() -> dict:
    """Read env on every reset so tests can monkeypatch between resets."""
    return {
        "models_dir": Path(os.environ.get("APP_MODELS_DIR", ROOT / "application" / "models")),
        "data_dir": Path(os.environ.get("APP_DATA_DIR", ROOT / "application" / "embeddings")),
        "default_model": os.environ.get("APP_DEFAULT_MODEL"),  # None → first loaded
    }


def _embedding_version(checkpoint: Path, name: str, use_tta: bool) -> str:
    h = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{name}-{h.hexdigest()[:16]}-tta{int(use_tta)}"


def _threshold_for(name: str, default: float) -> float:
    return float(os.environ.get(f"APP_THRESHOLD_{name.upper()}", default))


def _reset_for_tests() -> None:
    """(Re)load every model whose checkpoint exists. Missing → not loaded."""
    global _loaded, _default_model
    cfg = _config()
    _loaded = {}
    for name, spec in REGISTRY.items():
        ckpt = cfg["models_dir"] / spec.ckpt_filename
        if not ckpt.exists():
            continue
        embedder = Embedder(name, ckpt, device="cpu")
        aligner = FaceAligner(spec.input_size, spec.mean, spec.std, device="cpu")
        version = _embedding_version(ckpt, name, spec.use_tta)
        db = EnrollmentDB(cfg["data_dir"] / name, dim=embedder.dim, embedding_version=version)
        _loaded[name] = {
            "embedder": embedder, "aligner": aligner, "db": db,
            "threshold": _threshold_for(name, spec.threshold),
        }
    # Resolve default: explicit env if loaded, else first loaded model, else None.
    want = cfg["default_model"]
    _default_model = want if want in _loaded else (next(iter(_loaded), None))


class EnrollReq(BaseModel):
    name: str
    image: str
    model: str | None = None


class VerifyReq(BaseModel):
    image: str
    model: str | None = None
    threshold: float | None = None


def _resolve(model: str | None) -> tuple[str, dict]:
    """Resolve an optional model name to (name, loaded_entry), applying the default."""
    name = model if model is not None else _default_model
    if name is None:
        raise HTTPException(503, "no models available — drop a checkpoint in application/models/")
    if name not in REGISTRY:
        raise HTTPException(400, f"unknown model {name!r}")
    if name not in _loaded:
        raise HTTPException(503, f"model {name!r} unavailable — checkpoint missing")
    return name, _loaded[name]


def _decode_align_embed(b64: str, m: dict):
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "invalid base64 image")
    tensor = m["aligner"].align(img_bytes)
    if tensor is None:
        raise HTTPException(422, "no face detected")
    return m["embedder"].embed(tensor)


@app.get("/models")
def list_models() -> list[dict]:
    out = []
    for name, spec in REGISTRY.items():
        avail = name in _loaded
        out.append({
            "name": name, "available": avail, "input_size": spec.input_size,
            "threshold": _loaded[name]["threshold"] if avail else spec.threshold,
            "is_default": name == _default_model,
            "reason": None if avail else "checkpoint missing",
        })
    return out


@app.post("/enroll")
def enroll(req: EnrollReq) -> dict:
    name, m = _resolve(req.model)
    emb = _decode_align_embed(req.image, m)
    eid = m["db"].enroll(req.name, emb)
    return {"id": eid, "name": req.name, "model": name}


@app.post("/verify")
def verify(req: VerifyReq) -> dict:
    name, m = _resolve(req.model)
    emb = _decode_align_embed(req.image, m)
    thr = req.threshold if req.threshold is not None else m["threshold"]
    result = m["db"].verify(emb, threshold=thr)
    result["model"] = name
    return result


@app.get("/enrolled")
def list_enrolled(model: str | None = None) -> list[dict]:
    name = model if model is not None else _default_model
    if name not in _loaded:
        return []
    return _loaded[name]["db"].list_enrolled()


@app.delete("/enrolled/{eid}")
def delete_enrolled(eid: str, model: str | None = None) -> dict:
    name = model if model is not None else _default_model
    if name not in _loaded:
        return {"ok": False}
    _loaded[name]["db"].delete(eid)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND), check_dir=False), name="static")
