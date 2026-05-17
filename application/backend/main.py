"""FastAPI app: enroll / verify / list / delete endpoints + static frontend."""
from __future__ import annotations

import base64
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import EnrollmentDB
from .face_align import FaceAligner
from .inference import Embedder

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "application" / "frontend"

app = FastAPI(title="Face Recognition")

_aligner: FaceAligner | None = None
_embedder: Embedder | None = None
_db: EnrollmentDB | None = None
_threshold: float = 0.5


def _config() -> dict:
    return {
        "data_dir": Path(os.environ.get("APP_DATA_DIR", ROOT / "application" / "embeddings")),
        "checkpoint": Path(os.environ.get("APP_CKPT", ROOT / "application" / "models" / "best.pt")),
        "threshold": float(os.environ.get("APP_THRESHOLD", "0.5")),
    }


def _reset_for_tests() -> None:
    """Re-init globals from env. Lets tests set APP_DATA_DIR/APP_THRESHOLD per test."""
    global _aligner, _embedder, _db, _threshold
    cfg = _config()
    _aligner = FaceAligner(device="cpu")
    if cfg["checkpoint"].exists():
        _embedder = Embedder(cfg["checkpoint"], device="cpu")
        _db = EnrollmentDB(cfg["data_dir"], dim=_embedder.dim)
    else:
        _embedder = None
        _db = None
    _threshold = cfg["threshold"]


@app.on_event("startup")
def _startup() -> None:
    _reset_for_tests()


class EnrollReq(BaseModel):
    name: str
    image: str  # base64-encoded jpeg


class VerifyReq(BaseModel):
    image: str


def _decode_align_embed(b64: str):
    if _embedder is None or _db is None:
        raise HTTPException(
            503, "Model not loaded. Place checkpoint at application/models/best.pt."
        )
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "invalid base64 image")
    tensor = _aligner.align(img_bytes)
    if tensor is None:
        raise HTTPException(422, "no face detected")
    return _embedder.embed(tensor)


@app.post("/enroll")
def enroll(req: EnrollReq) -> dict:
    emb = _decode_align_embed(req.image)
    eid = _db.enroll(req.name, emb)
    return {"id": eid, "name": req.name}


@app.post("/verify")
def verify(req: VerifyReq) -> dict:
    emb = _decode_align_embed(req.image)
    return _db.verify(emb, threshold=_threshold)


@app.get("/enrolled")
def list_enrolled() -> list[dict]:
    if _db is None:
        return []
    return _db.list_enrolled()


@app.delete("/enrolled/{eid}")
def delete_enrolled(eid: str) -> dict:
    if _db is None:
        return {"ok": False}
    _db.delete(eid)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND), check_dir=False), name="static")
