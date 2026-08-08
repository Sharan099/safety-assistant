"""POST /regulations/upload + GET /regulations/jobs/{job_id}."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from api.ingest_worker import process_ingest_job
from api.jobs import create_job, get_job, init_db, set_job_pdf_path
from ingestion.detect_meta import detect_regulation_meta

logger = logging.getLogger(__name__)

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def _sanitize_filename(name: str) -> str:
    base = Path(name or "upload.pdf").name
    base = re.sub(r"[^\w.\-]+", "_", base)
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    return base[:180] or "upload.pdf"


@router.post("/regulations/upload")
async def upload_regulation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    regulation_id: str = Form(""),
    revision: str = Form(""),
):
    """Accept a PDF + optional ids; queue ingestion and return job_id immediately."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a .pdf file")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    init_db()

    raw_name = _sanitize_filename(file.filename)
    staging = RAW_DIR / f"_staging_{raw_name}"
    content = await file.read()
    if len(content) < 100:
        raise HTTPException(status_code=400, detail="PDF file is empty or too small")
    staging.write_bytes(content)

    rid = (regulation_id or "").strip()
    rev = (revision or "").strip()
    detected: dict[str, str] = {
        "regulation_id": "",
        "revision": "",
        "source": "none",
        "page1_preview": "",
    }
    if not rid or not rev:
        try:
            detected = detect_regulation_meta(staging)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cover detect failed: %s", exc)
        if not rid:
            rid = detected.get("regulation_id") or ""
        if not rev:
            rev = detected.get("revision") or ""

    if not rid:
        staging.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not detect regulation_id from the PDF cover. "
                "Please provide regulation_id (e.g. UN-ECE-R95)."
            ),
        )
    if not rev:
        rev = "Rev.unknown"

    job_id = create_job(
        pdf_path=str(staging),
        original_filename=file.filename or raw_name,
        regulation_id=rid,
        revision=rev,
    )
    final_path = RAW_DIR / f"{job_id}_{raw_name}"
    staging.replace(final_path)
    set_job_pdf_path(job_id, str(final_path), regulation_id=rid, revision=rev)

    background_tasks.add_task(_run_job_in_thread, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "regulation_id": rid,
        "revision": rev,
        "detected": {
            "regulation_id": detected.get("regulation_id") or "",
            "revision": detected.get("revision") or "",
            "source": detected.get("source") or "none",
        },
        "original_filename": file.filename,
    }


def _run_job_in_thread(job_id: str) -> None:
    """Sync wrapper for BackgroundTasks (runs in threadpool under Starlette)."""
    try:
        process_ingest_job(job_id)
    except Exception:  # noqa: BLE001
        logger.exception("background ingest crashed for %s", job_id)


@router.get("/regulations/jobs")
@router.get("/regulations/jobs/")
def list_ingest_jobs(limit: int = 20):
    """List recent ingest jobs (also absorbs empty `/regulations/jobs/` polls)."""
    from api.jobs import list_jobs

    rows = list_jobs(limit=max(1, min(limit, 100)))
    return {
        "jobs": [
            {
                "job_id": r["id"],
                "status": r["status"],
                "regulation_id": r.get("regulation_id") or "",
                "revision": r.get("revision") or "",
                "error": r.get("error"),
                "chunk_count": r.get("chunk_count"),
                "original_filename": r.get("original_filename") or "",
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
            }
            for r in rows
        ]
    }


@router.get("/regulations/jobs/{job_id}")
def get_ingest_job(job_id: str):
    """Poll ingestion job status."""
    cid = (job_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="job_id is required")
    job = get_job(cid)
    if not job:
        raise HTTPException(status_code=404, detail=f"Unknown job_id={cid!r}")
    return {
        "job_id": job["id"],
        "status": job["status"],
        "regulation_id": job.get("regulation_id") or "",
        "revision": job.get("revision") or "",
        "error": job.get("error"),
        "chunk_count": job.get("chunk_count"),
        "original_filename": job.get("original_filename") or "",
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


@router.post("/regulations/detect")
async def detect_meta_only(file: UploadFile = File(...)):
    """Peek cover page for regulation_id / revision without starting ingest."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a .pdf file")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RAW_DIR / f"_detect_{_sanitize_filename(file.filename)}"
    try:
        tmp.write_bytes(await file.read())
        return detect_regulation_meta(tmp)
    finally:
        tmp.unlink(missing_ok=True)
