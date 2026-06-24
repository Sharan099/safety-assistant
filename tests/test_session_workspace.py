"""Tests for per-session workspace isolation and paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.core.session_workspace import (
    WorkspacePaths,
    clear_session,
    get_workspace,
    load_manifest,
    upsert_manifest_doc,
    validate_session_id,
)
from backend.app.retrieval.hybrid import HybridRetriever
from config import CORPUS_MODE, SESSIONS_DIR


@pytest.fixture
def session_id() -> str:
    return "test_sess_001"


@pytest.fixture
def other_session_id() -> str:
    return "test_sess_002"


def test_validate_session_id_rejects_bad_ids():
    with pytest.raises(ValueError):
        validate_session_id("bad id!")
    with pytest.raises(ValueError):
        validate_session_id("short")


def test_workspace_paths_layout(session_id: str):
    ws = WorkspacePaths.for_session(session_id)
    assert ws.uploads == SESSIONS_DIR / session_id / "uploads"
    assert ws.chunks.name == "chunks.json"
    assert ws.manifest.name == "manifest.json"


def test_session_isolation_retrieval(session_id: str, other_session_id: str, tmp_path: Path, monkeypatch):
    sessions = tmp_path / "sessions"
    monkeypatch.setattr("config.SESSIONS_DIR", sessions)
    monkeypatch.setattr("backend.app.core.session_workspace.SESSIONS_DIR", sessions)
    monkeypatch.setattr("config.CORPUS_MODE", "session")

    ws_a = WorkspacePaths.for_session(session_id)
    ws_b = WorkspacePaths.for_session(other_session_id)
    ws_a.ensure_dirs()
    ws_b.ensure_dirs()

    chunks_a = {
        "chunks": [
            {
                "chunk_id": "a1",
                "text": "UN R94 chest deflection limit 42mm frontal",
                "pdf_name": "reg.pdf",
                "regulation": "UN_R94",
                "tier_confirmed": True,
            }
        ]
    }
    chunks_b = {
        "chunks": [
            {
                "chunk_id": "b1",
                "text": "measured chest deflection 38mm crash test",
                "pdf_name": "crash.pdf",
                "regulation": "PROG_X_FT_001",
                "doc_type": "test_report",
                "tier_confirmed": True,
            }
        ]
    }
    ws_a.chunks.parent.mkdir(parents=True, exist_ok=True)
    ws_a.chunks.write_text(json.dumps(chunks_a), encoding="utf-8")
    ws_b.chunks.parent.mkdir(parents=True, exist_ok=True)
    ws_b.chunks.write_text(json.dumps(chunks_b), encoding="utf-8")

    ret_a = HybridRetriever(chunks_file=ws_a.chunks, embeddings_file=ws_a.embeddings)
    ret_b = HybridRetriever(chunks_file=ws_b.chunks, embeddings_file=ws_b.embeddings)

    assert len(ret_a.chunks) == 1
    assert ret_a.chunks[0]["chunk_id"] == "a1"
    assert len(ret_b.chunks) == 1
    assert ret_b.chunks[0]["chunk_id"] == "b1"
    assert ret_a.chunks[0]["chunk_id"] != ret_b.chunks[0]["chunk_id"]

    clear_session(session_id)
    clear_session(other_session_id)


def test_unconfirmed_chunks_excluded_from_retriever(session_id: str, tmp_path: Path, monkeypatch):
    sessions = tmp_path / "sessions"
    monkeypatch.setattr("config.SESSIONS_DIR", sessions)
    monkeypatch.setattr("backend.app.core.session_workspace.SESSIONS_DIR", sessions)
    ws = WorkspacePaths.for_session(session_id)
    ws.ensure_dirs()
    data = {
        "chunks": [
            {"chunk_id": "x1", "text": "binding limit", "tier_confirmed": False},
            {"chunk_id": "x2", "text": "confirmed limit", "tier_confirmed": True},
        ]
    }
    ws.chunks.write_text(json.dumps(data), encoding="utf-8")
    ret = HybridRetriever(chunks_file=ws.chunks, embeddings_file=ws.embeddings)
    assert len(ret.chunks) == 1
    assert ret.chunks[0]["chunk_id"] == "x2"


def test_manifest_roundtrip(session_id: str, tmp_path: Path, monkeypatch):
    sessions = tmp_path / "sessions"
    monkeypatch.setattr("config.SESSIONS_DIR", sessions)
    monkeypatch.setattr("backend.app.core.session_workspace.SESSIONS_DIR", sessions)
    monkeypatch.setattr("config.CORPUS_MODE", "session")
    ws = get_workspace(session_id)
    ws.ensure_dirs()
    upsert_manifest_doc(
        ws,
        {
            "doc_id": "doc1",
            "filename": "UN_R94.pdf",
            "status": "pending_tier",
            "proposed_authority_tier": "legal_binding",
        },
    )
    manifest = load_manifest(ws)
    assert len(manifest["documents"]) == 1
    assert manifest["documents"][0]["filename"] == "UN_R94.pdf"
