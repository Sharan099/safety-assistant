"""Per-session upload workspaces — isolated folder trees for runtime ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    CHUNKS_FILE,
    CORPUS_DIR,
    CORPUS_MODE,
    EMBEDDINGS_FILE,
    INGEST_MANIFEST,
    MARKDOWN_DIR,
    OUTPUT_DIR,
    PAGE_IMAGE_CACHE,
    SESSIONS_DIR,
    SESSION_TTL_HOURS,
)

_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")


@dataclass(frozen=True)
class WorkspacePaths:
    """Filesystem layout for one ingestion workspace (session or local default)."""

    root: Path
    uploads: Path
    markdown: Path
    chunks: Path
    embeddings: Path
    bm25: Path
    manifest: Path
    page_cache: Path

    @classmethod
    def for_session(cls, session_id: str) -> WorkspacePaths:
        validate_session_id(session_id)
        root = SESSIONS_DIR / session_id
        return cls(
            root=root,
            uploads=root / "uploads",
            markdown=root / "markdown",
            chunks=root / "chunks" / "chunks.json",
            embeddings=root / "embeddings" / "embeddings.json",
            bm25=root / "bm25",
            manifest=root / "manifest.json",
            page_cache=root / "page_cache",
        )

    @classmethod
    def local_default(cls) -> WorkspacePaths:
        return cls(
            root=OUTPUT_DIR,
            uploads=CORPUS_DIR,
            markdown=MARKDOWN_DIR,
            chunks=CHUNKS_FILE,
            embeddings=EMBEDDINGS_FILE,
            bm25=OUTPUT_DIR / "bm25",
            manifest=INGEST_MANIFEST,
            page_cache=PAGE_IMAGE_CACHE,
        )

    def ensure_dirs(self) -> None:
        for d in (
            self.uploads,
            self.markdown,
            self.chunks.parent,
            self.embeddings.parent,
            self.bm25,
            self.page_cache,
        ):
            d.mkdir(parents=True, exist_ok=True)


def validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id or ""):
        raise ValueError("Invalid session_id")


def get_workspace(session_id: str | None = None) -> WorkspacePaths:
    if CORPUS_MODE == "local" or not session_id:
        return WorkspacePaths.local_default()
    return WorkspacePaths.for_session(session_id)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(ws: WorkspacePaths) -> dict[str, Any]:
    if not ws.manifest.exists():
        return {"documents": [], "session_id": ws.root.name, "updated_at": None}
    return json.loads(ws.manifest.read_text(encoding="utf-8"))


def save_manifest(ws: WorkspacePaths, manifest: dict[str, Any]) -> None:
    ws.ensure_dirs()
    manifest["updated_at"] = int(time.time())
    ws.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upsert_manifest_doc(ws: WorkspacePaths, doc: dict[str, Any]) -> None:
    manifest = load_manifest(ws)
    docs = manifest.get("documents", [])
    doc_id = doc["doc_id"]
    docs = [d for d in docs if d.get("doc_id") != doc_id]
    docs.append(doc)
    manifest["documents"] = docs
    if CORPUS_MODE == "session":
        manifest["session_id"] = ws.root.name
    save_manifest(ws, manifest)


def get_manifest_doc(ws: WorkspacePaths, doc_id: str) -> dict[str, Any] | None:
    for d in load_manifest(ws).get("documents", []):
        if d.get("doc_id") == doc_id:
            return d
    return None


def list_confirmed_docs(ws: WorkspacePaths) -> list[dict[str, Any]]:
    return [
        d
        for d in load_manifest(ws).get("documents", [])
        if d.get("status") == "ready" and d.get("tier_confirmed")
    ]


def session_has_indexed_docs(session_id: str) -> bool:
    ws = WorkspacePaths.for_session(session_id)
    return bool(list_confirmed_docs(ws)) and ws.chunks.exists()


def clear_session(session_id: str) -> dict[str, Any]:
    validate_session_id(session_id)
    ws = WorkspacePaths.for_session(session_id)
    if ws.root.exists():
        shutil.rmtree(ws.root)
    from backend.app.core.services import invalidate_retriever

    invalidate_retriever(session_id)
    return {"ok": True, "session_id": session_id}


def cleanup_expired_sessions() -> int:
    """Remove session trees older than SESSION_TTL_HOURS."""
    if not SESSIONS_DIR.exists():
        return 0
    cutoff = time.time() - SESSION_TTL_HOURS * 3600
    removed = 0
    for child in SESSIONS_DIR.iterdir():
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.json"
        mtime = manifest_path.stat().st_mtime if manifest_path.exists() else child.stat().st_mtime
        if mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed
