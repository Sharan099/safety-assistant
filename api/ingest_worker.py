"""Background worker: run staged regulation ingestion for a job id."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from api.cache_version import bump_cache_version
from api.jobs import get_job, update_job
from ingestion.chunk import chunk_document
from ingestion.embed_upsert import upsert_chunks
from ingestion.enrich import enrich_chunks
from ingestion.parse import parse_pdf, print_parse_summary
from retrieval.retrieve import invalidate_indexed_regulations_cache

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = Path(os.getenv("PDF_DIR") or ROOT / "data" / "pdfs")


def _safe_filename(regulation_id: str) -> str:
    rid = (regulation_id or "UNKNOWN").strip().upper().replace(" ", "-")
    short = rid.replace("UN-ECE-", "")
    return f"UN_{short}.pdf" if short.startswith("R") else f"{rid}.pdf"


def process_ingest_job(job_id: str) -> None:
    """Blocking ingest pipeline — call from a worker thread."""
    job = get_job(job_id)
    if not job:
        logger.error("ingest job missing: %s", job_id)
        return

    pdf = Path(job["pdf_path"])
    regulation_id = (job.get("regulation_id") or "").strip()
    revision = (job.get("revision") or "").strip() or "Rev.unknown"
    if not regulation_id:
        update_job(
            job_id,
            status="failed",
            error="regulation_id is required before ingestion",
        )
        return
    if not pdf.is_file():
        update_job(job_id, status="failed", error=f"PDF missing: {pdf}")
        return

    try:
        update_job(job_id, status="parsing", clear_error=True)
        export_dir = Path(os.getenv("DOCLING_EXPORT_DIR", str(ROOT / "data" / "docling")))
        doc = parse_pdf(pdf, export_dir=export_dir, do_ocr=False)
        print_parse_summary(doc)

        update_job(job_id, status="chunking")
        chunks = chunk_document(doc, regulation_id=regulation_id, revision=revision)
        enriched = enrich_chunks(chunks)

        update_job(job_id, status="embedding")
        n = upsert_chunks(enriched)

        # Publish under the standard PDF catalog path for GET /pdf/{id}.
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        dest = PDF_DIR / _safe_filename(regulation_id)
        if pdf.resolve() != dest.resolve():
            shutil.copy2(pdf, dest)

        bump_cache_version()
        invalidate_indexed_regulations_cache()

        update_job(job_id, status="done", chunk_count=n, clear_error=True)
        logger.info(
            "ingest job %s done regulation_id=%s chunks=%d",
            job_id,
            regulation_id,
            n,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingest job %s failed", job_id)
        update_job(job_id, status="failed", error=str(exc)[:2000])
