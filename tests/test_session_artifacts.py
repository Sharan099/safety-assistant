"""Tests for session artifact zip bundles."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from backend.app.core.session_artifacts import build_doc_artifact_zip, build_session_artifact_zip
from backend.app.core.session_workspace import WorkspacePaths, upsert_manifest_doc


@pytest.fixture
def session_id() -> str:
    return "artifact_sess01"


def test_session_artifact_zip_integrity(session_id: str, tmp_path: Path, monkeypatch):
    sessions = tmp_path / "sessions"
    monkeypatch.setattr("config.SESSIONS_DIR", sessions)
    monkeypatch.setattr("backend.app.core.session_workspace.SESSIONS_DIR", sessions)
    monkeypatch.setattr("config.CORPUS_MODE", "session")
    monkeypatch.setattr("backend.app.core.session_workspace.CORPUS_MODE", "session")
    ws = WorkspacePaths.for_session(session_id)
    ws.ensure_dirs()

    md_path = ws.markdown / "doc.md"
    md_path.write_text("# Test\n\nChest limit 42mm", encoding="utf-8")
    chunks = {
        "chunks": [
            {
                "chunk_id": "c1",
                "text": "CONTEXT: test\nChest limit 42mm",
                "pdf_name": "reg.pdf",
            }
        ]
    }
    ws.chunks.write_text(json.dumps(chunks), encoding="utf-8")
    upsert_manifest_doc(
        ws,
        {
            "doc_id": "d1",
            "filename": "reg.pdf",
            "markdown": "doc.md",
            "status": "ready",
            "tier_confirmed": True,
        },
    )

    blob = build_session_artifact_zip(session_id)
    with zipfile.ZipFile(BytesIO(blob)) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "chunks/chunks.json" in names
        assert "markdown/doc.md" in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["documents"][0]["doc_id"] == "d1"


def test_doc_artifact_zip(session_id: str, tmp_path: Path, monkeypatch):
    sessions = tmp_path / "sessions"
    monkeypatch.setattr("config.SESSIONS_DIR", sessions)
    monkeypatch.setattr("backend.app.core.session_workspace.SESSIONS_DIR", sessions)
    monkeypatch.setattr("config.CORPUS_MODE", "session")
    monkeypatch.setattr("backend.app.core.session_workspace.CORPUS_MODE", "session")
    ws = WorkspacePaths.for_session(session_id)
    ws.ensure_dirs()
    ws.markdown.mkdir(parents=True, exist_ok=True)
    (ws.markdown / "r.md").write_text("# R", encoding="utf-8")
    ws.chunks.write_text(
        json.dumps({"chunks": [{"chunk_id": "c1", "text": "t", "pdf_name": "r.pdf"}]}),
        encoding="utf-8",
    )
    upsert_manifest_doc(
        ws,
        {"doc_id": "d9", "filename": "r.pdf", "markdown": "r.md", "status": "ready"},
    )

    blob = build_doc_artifact_zip(session_id, "d9")
    with zipfile.ZipFile(BytesIO(blob)) as zf:
        assert "markdown/r.md" in zf.namelist()
        assert "chunks/d9_chunks.json" in zf.namelist()
