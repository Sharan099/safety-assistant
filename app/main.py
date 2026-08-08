import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from core.thread_config import configure_torch_threads

configure_torch_threads()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.config import settings
from api.routes import init_search_engine, router
from core.ingest import ingest_if_empty
from monitoring.metrics import prometheus_payload
from monitoring.request import RequestContextMiddleware


def _ingest_in_background() -> None:
    try:
        logger.info("AUTO_INGEST_ON_STARTUP=true — checking whether corpus ingest is needed")
        result = ingest_if_empty()
        if result is None:
            logger.info("Startup ingest skipped — corpus already present")
        else:
            logger.info("Startup ingest completed")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Startup ingest failed: {}", exc)


def _ping_gateway_in_background() -> None:
    try:
        from backend.app.gateway.gateway import ping_primary_provider

        ping_primary_provider()
        logger.info("Gateway primary provider reachable")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gateway startup check failed (continuing without blocking app startup): {}", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("SKIP_ML_STARTUP", "").lower() == "true":
        logger.info("SKIP_ML_STARTUP=true — ML models not preloaded at startup")
    else:
        logger.info("Preloading ML models at startup")
        init_search_engine()
        if os.getenv("SKIP_GATEWAY_STARTUP_CHECK", "").lower() != "true":
            threading.Thread(
                target=_ping_gateway_in_background,
                name="gateway-startup-check",
                daemon=True,
            ).start()
    if os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() == "true":
        threading.Thread(target=_ingest_in_background, name="startup-ingest", daemon=True).start()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="UNECE passive safety RAG assistant",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health",
        "ready": f"{settings.API_PREFIX}/ready",
    }


if settings.ENABLE_PROMETHEUS:

    @app.get("/metrics")
    async def metrics_root():
        return Response(content=prometheus_payload(), media_type="text/plain; version=0.0.4")
