"""Session artifact bundles — markdown, chunks, manifest for download."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from backend.app.core.session_workspace import (
    WorkspacePaths,
    get_workspace,
    load_manifest,
    validate_session_id,
)


def _chunks_for_doc(ws: WorkspacePaths, doc_id: str) -> list[dict]:
    from backend.app.core.session_workspace import get_manifest_doc

    doc = get_manifest_doc(ws, doc_id)
    if not doc or not ws.chunks.exists():
        return []
    pdf_name = doc.get("filename")
    data = json.loads(ws.chunks.read_text(encoding="utf-8"))
    return [c for c in data.get("chunks", []) if c.get("pdf_name") == pdf_name]


def build_doc_artifact_zip(session_id: str, doc_id: str) -> bytes:
    validate_session_id(session_id)
    ws = get_workspace(session_id)
    from backend.app.core.session_workspace import get_manifest_doc

    doc = get_manifest_doc(ws, doc_id)
    if not doc:
        raise FileNotFoundError(f"Document not found: {doc_id}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        md_name = doc.get("markdown")
        if md_name:
            md_path = ws.markdown / md_name
            if md_path.exists():
                zf.write(md_path, f"markdown/{md_name}")

        chunks = _chunks_for_doc(ws, doc_id)
        zf.writestr(
            f"chunks/{doc_id}_chunks.json",
            json.dumps({"chunks": chunks}, indent=2, ensure_ascii=False),
        )
        zf.writestr("manifest_doc.json", json.dumps(doc, indent=2, ensure_ascii=False))
    return buf.getvalue()


def build_session_artifact_zip(session_id: str) -> bytes:
    validate_session_id(session_id)
    ws = get_workspace(session_id)
    manifest = load_manifest(ws)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

        if ws.chunks.exists():
            zf.write(ws.chunks, "chunks/chunks.json")

        for md in sorted(ws.markdown.glob("*.md")):
            zf.write(md, f"markdown/{md.name}")

        for pdf in sorted(ws.uploads.glob("*.pdf")):
            zf.write(pdf, f"uploads/{pdf.name}")

    return buf.getvalue()


def list_session_documents(session_id: str) -> list[dict[str, Any]]:
    validate_session_id(session_id)
    ws = get_workspace(session_id)
    return load_manifest(ws).get("documents", [])
