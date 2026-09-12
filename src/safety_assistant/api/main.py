"""FastAPI entrypoint.

uv run uvicorn safety_assistant.api.main:app --port 8010
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from safety_assistant.api.middleware import RequestIdMiddleware
from safety_assistant.api.routes import admin, conversations, documents, health, me, query, versions
from safety_assistant.config import get_settings
from safety_assistant.observability import configure_tracing
from safety_assistant.observability.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
configure_tracing(app_env=settings.app_env)
log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Fail fast in production: the embedding model must load or the process must not serve."""
    if settings.app_env == "production":
        from safety_assistant.providers.embeddings import get_embedding_provider

        get_embedding_provider()
    log.info("startup complete", extra={"app_env": settings.app_env, "auth_mode": settings.auth_mode})
    yield


app = FastAPI(
    title="Safety Assistant — regulatory knowledge API",
    version="0.2.0",
    description="Versioned, auditable retrieval and grounded answers over automotive passive-safety regulations.",
    lifespan=_lifespan,
)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Requested-With"],
)
try:  # optional: auto-instrument HTTP spans when the OTel FastAPI instrumentor is installed
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health/live,health/ready,metrics")
except Exception:  # noqa: BLE001 — observability must never block startup
    log.warning("FastAPI OpenTelemetry instrumentation unavailable")
app.include_router(health.router)
app.include_router(query.router)
app.include_router(versions.router)
app.include_router(admin.router)
app.include_router(me.router)
app.include_router(conversations.router)
app.include_router(documents.router)
