"""Ingestion worker (ADR-0029 §5, D-003): PostgreSQL-backed queue, no broker.

    claim (FOR UPDATE SKIP LOCKED) → RUNNING → ingest_version → SUCCEEDED | FAILED (retry w/ backoff) | QUARANTINED

Idempotent: `ingest_version` re-runs from DISCOVERED and the version leaves the retrievable set until
re-verified; the partial unique index keeps one live job per version. Tests call `run_once` directly.

ponytail: single-table polling; upgrade path is a broker behind the same enqueue()/run_once() seam.
"""

from __future__ import annotations

import datetime as dt
import logging
import socket
import time
import uuid

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from safety_assistant.domain.documents import PUBLIC_ERRORS, classify_error
from safety_assistant.ingestion.fetch.blobstore import BlobStore
from safety_assistant.ingestion.workflows.ingest import IngestOutcome, ingest_version
from safety_assistant.observability import metrics
from safety_assistant.persistence import get_engine
from safety_assistant.persistence.models import IngestionJob
from safety_assistant.providers.embeddings import EmbeddingProvider
from safety_assistant.retrieval.sparse import invalidate_cache

log = logging.getLogger(__name__)

BACKOFF_MINUTES = (1, 5, 15)  # attempt 1 → +1 min, 2 → +5 min, 3 → +15 min


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def claim(session: Session, worker_id: str) -> IngestionJob | None:
    """Atomically take one due job; concurrent workers skip each other's rows."""
    # clock_timestamp(), not now(): now() is frozen at transaction start and a long-lived session
    # would never see jobs enqueued after it began.
    row = session.execute(
        text(
            "SELECT id FROM ingestion_jobs WHERE status = 'QUEUED' AND run_after <= clock_timestamp() "
            "ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED"
        )
    ).first()
    if row is None:
        session.rollback()  # do not hold a snapshot open while idle
        return None
    job = session.get(IngestionJob, row[0])
    assert job is not None
    job.status, job.locked_at, job.locked_by = "RUNNING", _now(), worker_id
    job.started_at = job.started_at or _now()
    job.attempt += 1
    session.commit()
    return job


def run_once(
    session: Session,
    *,
    worker_id: str | None = None,
    embedder: EmbeddingProvider | None = None,
    blob_store: BlobStore | None = None,
) -> IngestionJob | None:
    """Process at most one job. Returns it (in its final state) or None when the queue is empty."""
    worker_id = worker_id or f"{socket.gethostname()}:{uuid.uuid4().hex[:6]}"
    job = claim(session, worker_id)
    if job is None:
        return None
    started = time.perf_counter()
    outcome: IngestOutcome | None
    detail: str | None
    try:
        outcome = ingest_version(session, job.version_id, embedder=embedder, blob_store=blob_store, actor=worker_id)
        detail = outcome.error
    except Exception as exc:  # noqa: BLE001 — the job row must always reach a terminal/queued state
        log.exception("worker crashed on job %s", job.id)
        outcome, detail = None, f"{type(exc).__name__}: {exc}"
    session.rollback()  # ingest_version committed its own work; start clean for the job update
    job = session.get(IngestionJob, job.id)
    assert job is not None
    job.error_internal_ref = outcome.run_id if outcome else None
    status = outcome.status if outcome else "FAILED"
    if status in ("SUCCEEDED", "SKIPPED_UNCHANGED"):
        job.status, job.stage, job.completed_at = "SUCCEEDED", "READY", _now()
        job.error_code = job.error_public_message = None
        invalidate_cache()  # BM25 index must include the new version
    elif status == "QUARANTINED":
        job.status, job.completed_at = "QUARANTINED", _now()
        job.error_code = classify_error("QUARANTINED", detail)
        job.error_public_message = PUBLIC_ERRORS[job.error_code]
    else:
        job.error_code = classify_error("FAILED", detail)
        if job.attempt < job.max_attempts:
            job.status = "QUEUED"
            minutes = BACKOFF_MINUTES[min(job.attempt, len(BACKOFF_MINUTES)) - 1]
            job.run_after = func.now() + dt.timedelta(minutes=minutes)  # database clock, like the claim
            job.error_public_message = PUBLIC_ERRORS[job.error_code] + " Retry scheduled."
        else:
            job.status, job.completed_at = "FAILED", _now()
            job.error_code = "ATTEMPTS_EXHAUSTED"
            job.error_public_message = PUBLIC_ERRORS["ATTEMPTS_EXHAUSTED"]
    job.locked_at = job.locked_by = None
    session.commit()
    metrics.INGESTION_RUNS.labels(status=f"job_{job.status.lower()}").inc()
    log.info(
        "job finished",
        extra={
            "job_id": str(job.id),
            "status": job.status,
            "attempt": job.attempt,
            "seconds": round(time.perf_counter() - started, 1),
        },
    )
    return job


def run_forever(*, poll_seconds: float = 2.0, worker_id: str | None = None) -> None:
    worker_id = worker_id or f"{socket.gethostname()}:{uuid.uuid4().hex[:6]}"
    log.info("ingestion worker started", extra={"worker_id": worker_id})
    engine = get_engine()
    while True:
        with Session(engine, expire_on_commit=False) as session:
            job = run_once(session, worker_id=worker_id)
        if job is None:
            time.sleep(poll_seconds)


def queue_depth(session: Session) -> int:
    return len(session.scalars(select(IngestionJob.id).where(IngestionJob.status == "QUEUED")).all())
