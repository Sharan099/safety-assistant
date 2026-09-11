"""Privileged routes: ingestion, audit. Scope-checked before anything runs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from safety_assistant.api.dependencies import Principal, require_scope
from safety_assistant.ingestion.sources import get_registry
from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.persistence import get_session
from safety_assistant.persistence.models import IngestionEvent, IngestionRun, QueryTrace

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


class IngestRequest(BaseModel):
    source_key: str = Field(min_length=1, max_length=200)
    force: bool = False
    activate: bool = True


@router.post("/ingest")
async def ingest(
    req: IngestRequest,
    principal: Principal = Depends(require_scope("document:ingest")),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    registry = get_registry()
    if req.source_key not in registry.keys():
        raise HTTPException(404, "source_key is not in the registry allowlist")
    out = await run_in_threadpool(
        ingest_source,
        session,
        req.source_key,
        registry=registry,
        force=req.force,
        activate=req.activate,
        actor=principal.subject,
    )
    return {
        "run_id": str(out.run_id),
        "version_id": str(out.version_id) if out.version_id else None,
        "status": out.status,
        "version_status": out.final_version_status,
        "stats": out.stats,
        "error": out.error.splitlines()[0][:200] if out.error else None,
        "actor": principal.subject,
    }


@router.get("/ingestion/runs")
def runs(
    principal: Principal = Depends(require_scope("audit:read")),
    session: Session = Depends(get_session),
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = session.scalars(select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(min(limit, 500))).all()
    return [
        {
            "run_id": str(r.id),
            "source_key": r.source_key,
            "status": r.status,
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "attempt": r.attempt,
            "stats": r.stats,
            "error": r.error,
            "git_sha": r.git_sha,
        }
        for r in rows
    ]


@router.get("/ingestion/runs/{run_id}/events")
def run_events(
    run_id: str,
    principal: Principal = Depends(require_scope("audit:read")),
    session: Session = Depends(get_session),
) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(IngestionEvent).where(IngestionEvent.run_id == run_id).order_by(IngestionEvent.at)
    ).all()
    return [
        {
            "at": e.at,
            "from": e.from_status,
            "to": e.to_status,
            "level": e.level,
            "message": e.message,
            "payload": e.payload,
        }
        for e in rows
    ]


@router.get("/traces/{trace_id}")
def trace(
    trace_id: str,
    principal: Principal = Depends(require_scope("audit:read")),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    t = session.scalar(select(QueryTrace).where(QueryTrace.trace_id == trace_id))
    if t is None:
        raise HTTPException(404, "trace not found")
    return {
        "trace_id": t.trace_id, "created_at": t.created_at, "principal": t.principal, "query": t.query,
        "plan": t.plan, "candidates": t.candidates, "evidence": t.evidence, "answer": t.answer,
        "validation": t.validation, "versions": t.versions, "latency_ms": t.latency_ms, "tokens": t.tokens,
    }  # fmt: skip
