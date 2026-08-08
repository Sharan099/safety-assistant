"""SQLite job store for async regulation ingestion."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "jobs.sqlite3"

_lock = threading.Lock()
_db_path: Path | None = None

STATUSES = (
    "queued",
    "parsing",
    "chunking",
    "embedding",
    "done",
    "failed",
)


def _path() -> Path:
    global _db_path
    if _db_path is None:
        import os

        _db_path = Path(os.getenv("JOBS_DB") or DEFAULT_DB)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    return _db_path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    regulation_id TEXT,
                    revision TEXT,
                    pdf_path TEXT NOT NULL,
                    original_filename TEXT,
                    error TEXT,
                    chunk_count INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def create_job(
    *,
    pdf_path: str,
    original_filename: str = "",
    regulation_id: str = "",
    revision: str = "",
) -> str:
    init_db()
    job_id = uuid.uuid4().hex
    now = time.time()
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO ingest_jobs
                (id, status, regulation_id, revision, pdf_path, original_filename,
                 error, chunk_count, created_at, updated_at)
                VALUES (?, 'queued', ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    job_id,
                    regulation_id or "",
                    revision or "",
                    pdf_path,
                    original_filename or "",
                    now,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return job_id


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    regulation_id: str | None = None,
    revision: str | None = None,
    error: str | None = None,
    chunk_count: int | None = None,
    clear_error: bool = False,
) -> None:
    fields: list[str] = ["updated_at = ?"]
    values: list[Any] = [time.time()]
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if regulation_id is not None:
        fields.append("regulation_id = ?")
        values.append(regulation_id)
    if revision is not None:
        fields.append("revision = ?")
        values.append(revision)
    if chunk_count is not None:
        fields.append("chunk_count = ?")
        values.append(chunk_count)
    if clear_error:
        fields.append("error = NULL")
    elif error is not None:
        fields.append("error = ?")
        values.append(error)
    values.append(job_id)
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                f"UPDATE ingest_jobs SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()
        finally:
            conn.close()


def set_job_pdf_path(job_id: str, pdf_path: str, *, regulation_id: str, revision: str) -> None:
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                UPDATE ingest_jobs
                SET pdf_path = ?, regulation_id = ?, revision = ?, updated_at = ?
                WHERE id = ?
                """,
                (pdf_path, regulation_id, revision, time.time(), job_id),
            )
            conn.commit()
        finally:
            conn.close()


def get_job(job_id: str) -> dict[str, Any] | None:
    init_db()
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM ingest_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    return dict(row)


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """Most recent ingest jobs (newest first)."""
    init_db()
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM ingest_jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
    return [dict(r) for r in rows]
