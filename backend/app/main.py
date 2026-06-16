"""
AutoSafety RAG — FastAPI Backend
Architecture: API Gateway → FastAPI → LangGraph → Retriever → Reranker → LLM → Guardrails
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from backend.app.api.routes import router
from backend.app.core.security import SecurityHeadersMiddleware
from backend.app.core.settings import (
    API_PREFIX,
    CORS_ORIGINS,
    EXPOSE_GATEWAY_API,
)

app = FastAPI(
    title="AutoSafety RAG API",
    description="Passive safety regulation RAG with hybrid search, guardrails, "
    "and an Intelligent Multi-LLM Gateway",
    version="3.0.0",
)

# CORS: never combine wildcard origins with credentials (browser blocks it and
# it is a security risk). Use explicit origins when credentials are needed.
_origins = [o.strip() for o in CORS_ORIGINS if o.strip()]
_allow_credentials = "*" not in _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(router, prefix=API_PREFIX)

# OpenAI-compatible gateway endpoints (POST /api/v1/gateway/v1/chat/completions).
# Mounting the router is harmless when the gateway is disabled: the endpoints
# return 503 until ENABLE_GATEWAY=true.
if EXPOSE_GATEWAY_API:
    from backend.app.api.gateway_routes import router as gateway_router

    app.include_router(gateway_router, prefix=API_PREFIX)

# Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.on_event("startup")
async def startup() -> None:
    logger.info("Starting AutoSafety RAG backend...")
    try:
        from backend.app.core.services import run_self_test, warmup_pipeline

        await asyncio.to_thread(warmup_pipeline)
        # Run the end-to-end self-test in the background so startup is not
        # blocked; the frontend polls /ready until it passes.
        if os.getenv("RUN_SELFTEST_ON_STARTUP", "true").lower() == "true":
            asyncio.create_task(asyncio.to_thread(run_self_test))
    except Exception as exc:
        logger.error(f"Pipeline warmup failed: {exc}")


@app.get("/")
async def root() -> dict:
    return {
        "service": "autosafety-rag",
        "docs": "/docs",
        "api": API_PREFIX,
        "metrics": "/metrics",
    }
