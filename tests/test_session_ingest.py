"""Tests for ingest job state and authority-tier confirmation flow."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.core.session_ingest import confirm_authority_tier, get_job, start_upload
from backend.app.core.session_workspace import WorkspacePaths, get_manifest_doc


@pytest.fixture
def session_id() -> str:
    return "ingest_sess01"


def test_ingest_job_transitions_to_pending_tier(session_id: str, tmp_path: Path, monkeypatch):
    sessions = tmp_path / "sessions"
    monkeypatch.setattr("config.SESSIONS_DIR", sessions)
    monkeypatch.setattr("backend.app.core.session_workspace.SESSIONS_DIR", sessions)
    monkeypatch.setattr("config.CORPUS_MODE", "session")

    pdf_bytes = b"%PDF-1.4 minimal"

    def fake_pipeline(job_id, ws, pdf_path, doc_id):
        from backend.app.core import session_ingest as si

        si._set_job(job_id, status="pending_tier", stage="pending_tier", progress=60, doc_id=doc_id)
        from backend.app.core.session_workspace import upsert_manifest_doc

        upsert_manifest_doc(
            ws,
            {
                "doc_id": doc_id,
                "filename": pdf_path.name.split("_", 1)[-1],
                "status": "pending_tier",
                "proposed_authority_tier": "legal_binding",
                "tier_confirmed": False,
            },
        )

    with patch("backend.app.core.session_ingest._run_pipeline", side_effect=fake_pipeline):
        job_id = start_upload(session_id, pdf_bytes, "UN_R94.pdf")

    job = get_job(job_id)
    assert job is not None
    assert job["status"] == "pending_tier"
    assert job["stage"] == "pending_tier"


def test_authority_tier_confirm_updates_manifest(session_id: str, tmp_path: Path, monkeypatch):
    sessions = tmp_path / "sessions"
    monkeypatch.setattr("config.SESSIONS_DIR", sessions)
    monkeypatch.setattr("backend.app.core.session_workspace.SESSIONS_DIR", sessions)
    monkeypatch.setattr("config.CORPUS_MODE", "session")

    ws = WorkspacePaths.for_session(session_id)
    ws.ensure_dirs()
    ws.chunks.write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "text": "limit",
                        "pdf_name": "reg.pdf",
                        "tier_confirmed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    from backend.app.core.session_workspace import upsert_manifest_doc

    upsert_manifest_doc(
        ws,
        {
            "doc_id": "doc42",
            "filename": "reg.pdf",
            "status": "pending_tier",
            "proposed_authority_tier": "engineering_ref",
            "tier_confirmed": False,
        },
    )

    with patch("backend.app.core.session_ingest._run_embed_and_index"):
        result = confirm_authority_tier(session_id, "doc42", "legal_binding")

    assert result["authority_tier"] == "legal_binding"
    doc = get_manifest_doc(ws, "doc42")
    assert doc["tier_confirmed"] is True
    assert doc["authority_tier"] == "legal_binding"
    chunks = json.loads(ws.chunks.read_text(encoding="utf-8"))
    assert chunks["chunks"][0]["authority_tier"] == "legal_binding"
    assert chunks["chunks"][0]["tier_confirmed"] is True
