"""Verify embedding config matches live safety_registry.db."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from registry.embedding_config import EMBEDDING_DIMENSION, EMBEDDING_MODEL


@pytest.mark.skipif(
    not Path("safety_registry.db").exists(),
    reason="live DB not present",
)
def test_live_db_embedding_dimension_matches_config():
    conn = sqlite3.connect("safety_registry.db")
    row = conn.execute(
        "SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1"
    ).fetchone()
    assert row is not None
    vec = json.loads(row[0])
    assert len(vec) == EMBEDDING_DIMENSION == 768


def test_pinned_model_is_nomic():
    assert "nomic" in EMBEDDING_MODEL.lower()
    assert EMBEDDING_DIMENSION == 768
