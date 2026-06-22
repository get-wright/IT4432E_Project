from pathlib import Path

import numpy as np
import torch

from application.backend.db import EnrollmentDB


def test_enroll_verify_delete_roundtrip(tmp_path: Path):
    db = EnrollmentDB(tmp_path, dim=8)
    id1 = db.enroll(
        "alice",
        torch.tensor(np.array([1., 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
    )
    id2 = db.enroll(
        "bob",
        torch.tensor(np.array([0., 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
    )
    assert id1 != id2

    query = torch.tensor(np.array([0.99, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    query = query / query.norm()
    result = db.verify(query, threshold=0.9)
    assert result["matched"] is True
    assert result["best_match"] == "alice"
    assert result["score"] > 0.9

    enrolled = db.list_enrolled()
    assert {e["name"] for e in enrolled} == {"alice", "bob"}

    db.delete(id2)
    assert {e["name"] for e in db.list_enrolled()} == {"alice"}

    # Persist + reload from disk.
    db2 = EnrollmentDB(tmp_path, dim=8)
    assert {e["name"] for e in db2.list_enrolled()} == {"alice"}


def test_verify_empty_returns_no_match(tmp_path: Path):
    db = EnrollmentDB(tmp_path, dim=8)
    q = torch.zeros(8)
    q[0] = 1.0
    result = db.verify(q, threshold=0.5)
    assert result["matched"] is False
    assert result["best_match"] is None


def test_delete_by_name_removes_all_matching_rows(tmp_path: Path):
    db = EnrollmentDB(tmp_path, dim=8)
    eye = np.eye(8, dtype=np.float32)
    db.enroll("alice", torch.tensor(eye[0]))
    db.enroll("alice", torch.tensor(eye[1]))  # same name twice
    db.enroll("bob", torch.tensor(eye[2]))

    removed = db.delete_by_name("alice")
    assert removed == 2
    assert {e["name"] for e in db.list_enrolled()} == {"bob"}

    # bob's vector survived compaction and still verifies.
    res = db.verify(torch.tensor(eye[2]), threshold=0.9)
    assert res["best_match"] == "bob"

    # Persist + reload from disk.
    db2 = EnrollmentDB(tmp_path, dim=8)
    assert {e["name"] for e in db2.list_enrolled()} == {"bob"}

    # Deleting an absent name is a no-op returning 0.
    assert db.delete_by_name("nobody") == 0

