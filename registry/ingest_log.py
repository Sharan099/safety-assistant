"""Per-stage ingest observability (NFR-4)."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.orm import Session

from database.models import IngestLog


def new_run_id() -> str:
    return uuid.uuid4().hex


def log_stage(
    db: Session,
    *,
    run_id: str,
    stage: str,
    item: str,
    outcome: str,
    reason: str | None = None,
    duration_ms: int = 0,
) -> None:
    db.add(
        IngestLog(
            run_id=run_id,
            stage=stage,
            item=item,
            outcome=outcome,
            reason=reason,
            duration_ms=duration_ms,
        )
    )
    db.commit()


@contextmanager
def timed_stage(
    db: Session,
    *,
    run_id: str,
    stage: str,
    item: str,
) -> Iterator[dict]:
    started = time.perf_counter()
    ctx: dict = {"outcome": "ok", "reason": None}
    try:
        yield ctx
    except Exception as exc:
        ctx["outcome"] = "error"
        ctx["reason"] = str(exc)
        raise
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        log_stage(
            db,
            run_id=run_id,
            stage=stage,
            item=item,
            outcome=ctx.get("outcome", "ok"),
            reason=ctx.get("reason"),
            duration_ms=duration_ms,
        )
