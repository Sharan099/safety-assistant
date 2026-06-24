"""Session-scoped async document ingestion with per-stage progress."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from backend.app.core.session_workspace import (
    WorkspacePaths,
    file_sha256,
    get_workspace,
    upsert_manifest_doc,
    validate_session_id,
)
from config import CORPUS_MODE

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

STAGES = (
    "queued",
    "extracting",
    "chunking",
    "classifying",
    "pending_tier",
    "embedding",
    "indexing",
    "ready",
    "failed",
)

_TIER_MAP = {
    "legal_binding": "legal_binding",
    "rating_protocol": "rating_protocol",
    "engineering_ref": "engineering_ref",
    "oem_internal": "oem_internal",
    "historical_data": "historical_data",
    "legal": "legal_binding",
    "rating": "rating_protocol",
    "reference": "engineering_ref",
    "internal": "oem_internal",
}


def _set_job(job_id: str, **kwargs: Any) -> None:
    with _LOCK:
        _JOBS.setdefault(job_id, {})
        _JOBS[job_id].update(kwargs)
        _JOBS[job_id]["updated_at"] = time.time()


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        return _JOBS.get(job_id, {}).copy() if job_id in _JOBS else None


def list_session_jobs(session_id: str) -> list[dict[str, Any]]:
    with _LOCK:
        return [j.copy() for j in _JOBS.values() if j.get("session_id") == session_id]


def _normalize_tier(raw: str | None) -> str:
    if not raw:
        return "engineering_ref"
    key = raw.strip().lower()
    return _TIER_MAP.get(key, key if key in _TIER_MAP.values() else "engineering_ref")


def _merge_chunks(ws: WorkspacePaths, new_chunks: list[dict], pdf_name: str) -> list[dict]:
    existing: list[dict] = []
    if ws.chunks.exists():
        data = json.loads(ws.chunks.read_text(encoding="utf-8"))
        existing = [c for c in data.get("chunks", []) if c.get("pdf_name") != pdf_name]
    return existing + new_chunks


def _write_chunks(ws: WorkspacePaths, all_chunks: list[dict]) -> None:
    ws.chunks.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": "session_hierarchical",
        "total_chunks": len(all_chunks),
        "unique_chunk_ids": len({c["chunk_id"] for c in all_chunks}),
        "chunks": all_chunks,
        "metadata_schema_version": 2,
    }
    ws.chunks.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_embed_and_index(ws: WorkspacePaths, job_id: str, doc_id: str) -> None:
    from ingestion.embed_chunks import run_for_paths

    _set_job(job_id, status="embedding", stage="embedding", progress=75)
    run_for_paths(ws.chunks, ws.embeddings)
    _set_job(job_id, status="indexing", stage="indexing", progress=90)
    from backend.app.core.services import invalidate_retriever

    invalidate_retriever(ws.root.name)
    upsert_manifest_doc(
        ws,
        {
            **(_manifest_entry(ws, doc_id) or {}),
            "status": "ready",
            "stage": "ready",
            "progress": 100,
        },
    )
    _set_job(job_id, status="ready", stage="ready", progress=100)


def _manifest_entry(ws: WorkspacePaths, doc_id: str) -> dict[str, Any] | None:
    from backend.app.core.session_workspace import get_manifest_doc

    return get_manifest_doc(ws, doc_id)


def _run_pipeline(job_id: str, ws: WorkspacePaths, pdf_path: Path, doc_id: str) -> None:
    filename = pdf_path.name
    try:
        _set_job(job_id, status="extracting", stage="extracting", progress=15)
        from ingestion.docling_converter import convert_single_pdf

        result = convert_single_pdf(
            pdf_path,
            force=True,
            markdown_dir=ws.markdown,
            page_cache_dir=ws.page_cache,
        )
        md_name = result.get("markdown", pdf_path.stem + ".md")
        md_path = ws.markdown / md_name

        _set_job(job_id, status="chunking", stage="chunking", progress=40)
        from ingestion.hierarchical_chunker import chunk_markdown_file

        new_chunks = chunk_markdown_file(md_path)
        all_chunks = _merge_chunks(ws, new_chunks, filename)
        _write_chunks(ws, all_chunks)

        _set_job(job_id, status="classifying", stage="classifying", progress=55)
        from ingestion.hierarchical_chunker import detect_regulation_type
        from ingestion.metadata_classifier import classify_document_defaults

        regulation = detect_regulation_type(filename)
        defaults = classify_document_defaults(regulation, filename)
        proposed_tier = _normalize_tier(defaults.get("authority_tier"))
        page_count = defaults.get("page_count") or 0

        doc_entry = {
            "doc_id": doc_id,
            "filename": filename,
            "sha256": file_sha256(pdf_path),
            "doc_type": defaults.get("doc_type", "reference"),
            "proposed_authority_tier": proposed_tier,
            "authority_tier": proposed_tier,
            "tier_confirmed": False,
            "region": defaults.get("region", "global"),
            "impact_mode": defaults.get("impact_mode", "general"),
            "regulation": regulation,
            "page_count": page_count,
            "chunk_count": len(new_chunks),
            "status": "pending_tier",
            "stage": "pending_tier",
            "progress": 60,
            "markdown": md_name,
        }
        upsert_manifest_doc(ws, doc_entry)
        _set_job(
            job_id,
            status="pending_tier",
            stage="pending_tier",
            progress=60,
            doc_id=doc_id,
            proposed_authority_tier=proposed_tier,
            authority_tier=proposed_tier,
            tier_confirmed=False,
            chunk_count=len(new_chunks),
        )
    except Exception as exc:
        upsert_manifest_doc(
            ws,
            {
                "doc_id": doc_id,
                "filename": filename,
                "status": "failed",
                "stage": "failed",
                "error": str(exc),
            },
        )
        _set_job(job_id, status="failed", stage="failed", progress=100, error=str(exc))


def start_upload(
    session_id: str,
    file_bytes: bytes,
    filename: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    if CORPUS_MODE != "session":
        raise RuntimeError("Session upload requires CORPUS_MODE=session")
    validate_session_id(session_id)
    ws = get_workspace(session_id)
    ws.ensure_dirs()

    doc_id = str(uuid.uuid4())[:12]
    job_id = str(uuid.uuid4())
    safe_name = Path(filename).name
    pdf_path = ws.uploads / f"{doc_id}_{safe_name}"
    pdf_path.write_bytes(file_bytes)

    meta = metadata or {}
    _set_job(
        job_id,
        session_id=session_id,
        doc_id=doc_id,
        status="queued",
        stage="queued",
        progress=5,
        pdf_name=safe_name,
        filename=safe_name,
        metadata=meta,
        created_at=time.time(),
    )

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, ws, pdf_path, doc_id),
        daemon=True,
    )
    thread.start()
    return job_id


def confirm_authority_tier(
    session_id: str,
    doc_id: str,
    authority_tier: str,
    *,
    job_id: str | None = None,
) -> dict[str, Any]:
    """User confirms or overrides proposed authority tier; then embed + index."""
    validate_session_id(session_id)
    ws = get_workspace(session_id)
    from backend.app.core.session_workspace import get_manifest_doc

    doc = get_manifest_doc(ws, doc_id)
    if not doc:
        raise ValueError(f"Document not found: {doc_id}")
    if doc.get("status") == "ready" and doc.get("tier_confirmed"):
        return {"ok": True, "doc_id": doc_id, "already_confirmed": True}

    tier = _normalize_tier(authority_tier)
    doc["authority_tier"] = tier
    doc["tier_confirmed"] = True

    if ws.chunks.exists():
        data = json.loads(ws.chunks.read_text(encoding="utf-8"))
        pdf_name = doc.get("filename")
        for c in data.get("chunks", []):
            if c.get("pdf_name") == pdf_name:
                c["authority_tier"] = tier
                c["tier_confirmed"] = True
        _write_chunks(ws, data.get("chunks", []))

    upsert_manifest_doc(ws, {**doc, "status": "embedding", "stage": "embedding", "progress": 70})

    resolved_job = job_id
    if not resolved_job:
        for j in list_session_jobs(session_id):
            if j.get("doc_id") == doc_id:
                resolved_job = j.get("job_id")
                break

    if resolved_job:
        _set_job(
            resolved_job,
            status="embedding",
            stage="embedding",
            progress=70,
            authority_tier=tier,
            tier_confirmed=True,
        )
        thread = threading.Thread(
            target=_run_embed_and_index,
            args=(ws, resolved_job, doc_id),
            daemon=True,
        )
        thread.start()
    else:
        fallback_job = f"confirm-{doc_id}"
        _set_job(
            fallback_job,
            status="embedding",
            stage="embedding",
            progress=70,
            doc_id=doc_id,
            session_id=session_id,
            authority_tier=tier,
            tier_confirmed=True,
        )
        thread = threading.Thread(
            target=_run_embed_and_index,
            args=(ws, fallback_job, doc_id),
            daemon=True,
        )
        thread.start()

    return {"ok": True, "doc_id": doc_id, "authority_tier": tier, "status": "embedding"}
