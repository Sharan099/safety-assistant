"""Health semantics (CLAUDE.md §15):

- /health/live   process alive;
- /health/ready  safe to receive traffic: database reachable, migrations at head,
                 embedding provider loaded, at least one ACTIVE version. The LLM is
                 *not* a readiness dependency — evidence-only is an accepted mode —
                 but its status is reported under /health/deps.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from safety_assistant.config import get_settings
from safety_assistant.domain.regulations import VersionStatus
from safety_assistant.observability import metrics
from safety_assistant.persistence import get_engine
from safety_assistant.persistence.models import RegulationVersion
from safety_assistant.providers.embeddings import get_embedding_provider
from safety_assistant.providers.llm import get_llm_provider

router = APIRouter(tags=["health"])


def _db_check() -> dict[str, Any]:
    with Session(get_engine()) as s:
        s.execute(text("SELECT 1"))
        head = s.execute(text("SELECT version_num FROM alembic_version")).scalar()
        active = s.scalar(
            select(func.count())
            .select_from(RegulationVersion)
            .where(RegulationVersion.status == VersionStatus.ACTIVE.value)
        )
    return {"ok": True, "migration": head, "active_versions": int(active or 0)}


def _embedding_check() -> dict[str, Any]:
    p = get_embedding_provider()
    return {"ok": True, "model": p.model_name, "dimensions": p.dimensions}


def _llm_check() -> dict[str, Any]:
    p = get_llm_provider()
    return {"ok": p is not None, "provider": getattr(p, "name", None), "model": getattr(p, "model", None)}


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(response: Response) -> dict[str, Any]:
    deps: dict[str, Any] = {}
    for name, fn in (("database", _db_check), ("embeddings", _embedding_check)):
        try:
            deps[name] = await run_in_threadpool(fn)
        except Exception as exc:  # noqa: BLE001 — readiness reports, never crashes
            deps[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    ready_ok = all(d.get("ok") for d in deps.values()) and deps["database"].get("active_versions", 0) > 0
    if not ready_ok:
        response.status_code = 503
    return {"status": "ready" if ready_ok else "not_ready", "app_env": get_settings().app_env, "deps": deps}


@router.get("/health/deps")
async def dependencies() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fn in (("database", _db_check), ("embeddings", _embedding_check), ("llm", _llm_check)):
        try:
            out[name] = await run_in_threadpool(fn)
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    out["llm"].setdefault("note", "LLM outage degrades to evidence-only mode; not a readiness dependency")
    return out


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> PlainTextResponse:
    """Prometheus scrape endpoint. Expose on the internal listener only (see infra/)."""
    body, content_type = metrics.render()
    return PlainTextResponse(content=body, media_type=content_type)
