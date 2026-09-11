"""FastAPI entrypoint.

uv run uvicorn safety_assistant.api.main:app --port 8010
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from safety_assistant.api.middleware import RequestIdMiddleware
from safety_assistant.api.routes import admin, health, query
from safety_assistant.config import get_settings
from safety_assistant.observability.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Safety Assistant — regulatory knowledge API",
    version="0.2.0",
    description="Versioned, auditable retrieval and grounded answers over automotive passive-safety regulations.",
)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)
app.include_router(health.router)
app.include_router(query.router)
app.include_router(admin.router)


@app.on_event("startup")
def _warm_up() -> None:
    """Fail fast in production: the embedding model must load or the process must not serve."""
    if settings.app_env == "production":
        from safety_assistant.providers.embeddings import get_embedding_provider

        get_embedding_provider()
    log.info("startup complete", extra={"app_env": settings.app_env, "auth_mode": settings.auth_mode})
