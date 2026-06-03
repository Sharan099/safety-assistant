"""
AutoSafety RAG — FastAPI Backend
Architecture: API Gateway → FastAPI → LangGraph → Retriever → Reranker → LLM → Guardrails
"""

import asyncio
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
from backend.app.core.settings import API_PREFIX, CORS_ORIGINS

app = FastAPI(
    title="AutoSafety RAG API",
    description="Passive safety regulation RAG with hybrid search and guardrails",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=API_PREFIX)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.on_event("startup")
async def startup() -> None:
    logger.info("Starting AutoSafety RAG backend...")
    try:
        from backend.app.core.services import warmup_pipeline

        await asyncio.to_thread(warmup_pipeline)
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
