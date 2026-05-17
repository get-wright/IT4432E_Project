"""SQLite + .npy enrollment storage with cosine-similarity verify."""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import torch


class EnrollmentDB:
    def __init__(self, root: Path, dim: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        self.vec_path = self.root / "vectors.npy"
        self.dim = dim
        self._init_db()
        self._load_vectors()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS enrolled (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    row_idx INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )

    def _load_vectors(self) -> None:
        if self.vec_path.exists():
            arr = np.load(self.vec_path)
            if arr.ndim == 1:
                arr = arr.reshape(0, self.dim)
        else:
            arr = np.zeros((0, self.dim), dtype=np.float32)
        self._vectors = arr

    def _save_vectors(self) -> None:
        np.save(self.vec_path, self._vectors)

    def _compact(self) -> None:
        """Rebuild vectors.npy + row_idx to remove deleted entries."""
        with sqlite3.connect(self.db_path) as c:
            rows = list(c.execute("SELECT id, row_idx FROM enrolled ORDER BY row_idx"))
            if rows:
                self._vectors = np.stack([self._vectors[r[1]] for r in rows])
            else:
                self._vectors = np.zeros((0, self.dim), dtype=np.float32)
            for new_idx, (eid, _) in enumerate(rows):
                c.execute("UPDATE enrolled SET row_idx=? WHERE id=?", (new_idx, eid))
        self._save_vectors()

    def enroll(self, name: str, emb: torch.Tensor) -> str:
        emb_np = emb.detach().cpu().numpy().astype(np.float32)
        assert emb_np.shape == (self.dim,), f"expected ({self.dim},), got {emb_np.shape}"
        row_idx = self._vectors.shape[0]
        self._vectors = np.vstack([self._vectors, emb_np[None, :]])
        self._save_vectors()
        eid = uuid.uuid4().hex
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO enrolled(id,name,row_idx,created_at) VALUES (?,?,?,?)",
                (eid, name, row_idx, time.time()),
            )
        return eid

    def delete(self, eid: str) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute("DELETE FROM enrolled WHERE id=?", (eid,))
        self._compact()

    def list_enrolled(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as c:
            return [
                {"id": r[0], "name": r[1], "created_at": r[2]}
                for r in c.execute(
                    "SELECT id, name, created_at FROM enrolled ORDER BY created_at"
                )
            ]

    def verify(self, emb: torch.Tensor, threshold: float) -> dict:
        if self._vectors.shape[0] == 0:
            return {"matched": False, "best_match": None, "score": 0.0, "threshold": threshold}
        q = emb.detach().cpu().numpy().astype(np.float32)
        sims = self._vectors @ q  # both L2-normalized -> cosine.
        with sqlite3.connect(self.db_path) as c:
            rows = list(c.execute("SELECT name, row_idx FROM enrolled"))
        best_score = -1.0
        best_name: str | None = None
        for name, idx in rows:
            s = float(sims[idx])
            if s > best_score:
                best_score, best_name = s, name
        return {
            "matched": best_score >= threshold,
            "best_match": best_name,
            "score": best_score,
            "threshold": threshold,
        }
