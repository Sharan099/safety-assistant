"""Document upload / ingest / delete service."""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from config import CHUNKS_FILE, CORPUS_DIR, EMBEDDINGS_FILE, MARKDOWN_DIR

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

REFERENCE = CORPUS_DIR / "reference"
LEGAL = CORPUS_DIR / "legal"
RATING = CORPUS_DIR / "rating"


def _corpus_dir_for_doc_type(doc_type: str) -> Path:
    if doc_type == "legal":
        return LEGAL
    if doc_type == "rating":
        return RATING
    return REFERENCE


def list_documents() -> list[dict[str, Any]]:
  docs: list[dict[str, Any]] = []
  manifest = Path(CORPUS_DIR).parent / "manifest" / "corpus_manifest.json"
  if manifest.exists():
      data = json.loads(manifest.read_text(encoding="utf-8"))
      for d in data.get("documents", []):
          docs.append(d)
  else:
      for pdf in CORPUS_DIR.rglob("*.pdf"):
          docs.append({
              "path": str(pdf),
              "name": pdf.name,
              "category": pdf.parent.name,
          })
  return docs


def _set_job(job_id: str, **kwargs: Any) -> None:
    with _LOCK:
        _JOBS.setdefault(job_id, {})
        _JOBS[job_id].update(kwargs)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        return _JOBS.get(job_id)


def _run_ingest(job_id: str, pdf_path: Path, metadata: dict[str, Any]) -> None:
    try:
        _set_job(job_id, status="ocr", progress=10)
        from ingestion.docling_converter import convert_single_pdf
        from ingestion.quality_gate import check_markdown

        result = convert_single_pdf(pdf_path, force=True)
        md_name = result.get("markdown", pdf_path.stem + ".md")
        md_path = MARKDOWN_DIR / md_name

        _set_job(job_id, status="quality_gate", progress=30)
        qr = check_markdown(md_path)
        if not qr.passed:
            _set_job(
                job_id,
                status="failed",
                progress=100,
                error=f"Quality gate failed: {qr.issues} (page: {qr.failed_page})",
            )
            return

        _set_job(job_id, status="chunking", progress=50)
        from ingestion.hierarchical_chunker import chunk_markdown_file, run as run_chunk

        new_chunks = chunk_markdown_file(md_path)
        if CHUNKS_FILE.exists():
            data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
            existing = [c for c in data.get("chunks", []) if c.get("pdf_name") != pdf_path.name]
            all_chunks = existing + new_chunks
        else:
            all_chunks = new_chunks
        data = {
            "pipeline": "docling_hierarchical",
            "total_chunks": len(all_chunks),
            "unique_chunk_ids": len({c["chunk_id"] for c in all_chunks}),
            "chunks": all_chunks,
            "metadata_schema_version": 2,
        }
        CHUNKS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        _set_job(job_id, status="embedding", progress=70)
        from ingestion.embed_chunks import run as run_embed

        run_embed()

        _set_job(
            job_id,
            status="complete",
            progress=100,
            chunk_count=len(new_chunks),
            pdf_name=pdf_path.name,
        )
    except Exception as exc:
        _set_job(job_id, status="failed", progress=100, error=str(exc))


def start_upload(
    file_bytes: bytes,
    filename: str,
    metadata: dict[str, Any],
) -> str:
    job_id = str(uuid.uuid4())
    doc_type = metadata.get("doc_type", "reference")
    dest_dir = _corpus_dir_for_doc_type(doc_type)
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = dest_dir / filename
    pdf_path.write_bytes(file_bytes)

    _set_job(
        job_id,
        status="uploaded",
        progress=5,
        pdf_name=filename,
        metadata=metadata,
        created_at=time.time(),
    )
    thread = threading.Thread(
        target=_run_ingest,
        args=(job_id, pdf_path, metadata),
        daemon=True,
    )
    thread.start()
    return job_id


def delete_document(doc_id: str) -> dict[str, Any]:
    """Remove PDF, markdown, chunks, and embeddings for a document."""
    removed_chunks = 0
    pdf_path: Path | None = None
    for pdf in CORPUS_DIR.rglob("*.pdf"):
        if doc_id.lower() in pdf.name.lower() or doc_id.upper() in pdf.stem.upper():
            pdf_path = pdf
            break
    if not pdf_path:
        return {"ok": False, "error": f"Document not found: {doc_id}"}

    md_path = MARKDOWN_DIR / (pdf_path.stem + ".md")
    if md_path.exists():
        md_path.unlink()

    if CHUNKS_FILE.exists():
        data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        chunks = data.get("chunks", [])
        keep = [c for c in chunks if c.get("pdf_name") != pdf_path.name]
        removed_chunks = len(chunks) - len(keep)
        remove_ids = {c["chunk_id"] for c in chunks if c.get("pdf_name") == pdf_path.name}
        data["chunks"] = keep
        data["total_chunks"] = len(keep)
        CHUNKS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        if EMBEDDINGS_FILE.exists():
            edata = json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
            embs = edata.get("embeddings", {})
            for cid in remove_ids:
                embs.pop(cid, None)
            edata["embeddings"] = embs
            EMBEDDINGS_FILE.write_text(json.dumps(edata), encoding="utf-8")

    pdf_path.unlink()
    return {"ok": True, "removed_chunks": removed_chunks, "pdf": pdf_path.name}
