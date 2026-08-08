"""Jobs store + cache version + cover detect heuristics."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.cache_version import bump_cache_version, get_cache_version
from api.jobs import create_job, get_job, init_db, update_job
from generation.answer_cache import cache_key
from ingestion.detect_meta import detect_from_text


@pytest.fixture()
def jobs_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "jobs.sqlite3"
    monkeypatch.setenv("JOBS_DB", str(db))
    monkeypatch.setattr("api.jobs._db_path", None)
    init_db()
    return db


def test_job_lifecycle(jobs_db: Path):
    jid = create_job(pdf_path="/tmp/x.pdf", original_filename="x.pdf", regulation_id="UN-ECE-R95")
    row = get_job(jid)
    assert row is not None
    assert row["status"] == "queued"
    update_job(jid, status="parsing")
    update_job(jid, status="chunking")
    update_job(jid, status="embedding")
    update_job(jid, status="done", chunk_count=42)
    done = get_job(jid)
    assert done["status"] == "done"
    assert done["chunk_count"] == 42


def test_cache_version_changes_answer_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ver = tmp_path / "cache_version.json"
    monkeypatch.setattr("api.cache_version.VERSION_PATH", ver)
    v0 = get_cache_version()
    k0 = cache_key("Which vehicles are covered under UN R95?", regulation_id="UN-ECE-R95")
    bump_cache_version()
    k1 = cache_key("Which vehicles are covered under UN R95?", regulation_id="UN-ECE-R95")
    assert get_cache_version() == v0 + 1
    assert k0 != k1


def test_detect_r95_from_cover_text():
    text = "Agreement ... Uniform provisions ... Regulation No. 95 Revision 3 lateral collision"
    rid, rev = detect_from_text(text)
    assert rid == "UN-ECE-R95"
    assert rev.startswith("Rev.")
