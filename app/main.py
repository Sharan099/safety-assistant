import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load repo-root .env so gateway providers (GROQ_API_KEY, etc.) see os.getenv values.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from api.routes import router as registry_router
from api.auth_routes import router as auth_router
from api.user_upload_routes import router as user_upload_router
from scheduler.jobs import start_scheduler

# Initialize FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    description="Regulatory Knowledge Infrastructure for Passive Safety Engineering",
    version="1.0.0"
)

# Configure CORS Middleware — never combine wildcard origins with credentials.
_registry_origins = os.getenv(
    "REGISTRY_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8002,http://localhost:8080",
).split(",")
_cors_origins = [o.strip() for o in _registry_origins if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(auth_router, prefix=settings.API_PREFIX)
app.include_router(user_upload_router, prefix=settings.API_PREFIX)
app.include_router(registry_router, prefix=settings.API_PREFIX)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Regulation Registry App Startup Actions...")
    
    # 1. Start APScheduler jobs (for crawlers and scheduled tasks)
    if os.getenv("DISABLE_SCHEDULER", "false").lower() != "true":
        try:
            start_scheduler()
        except Exception as e:
            logger.error(f"Startup: Failed to initialize APScheduler: {e}")
    else:
        logger.info("APScheduler disabled (DISABLE_SCHEDULER=true).")
        
    logger.info("Regulation Registry App successfully started.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Regulation Registry App...")


# Setup Prometheus Monitoring instrumentation
if settings.ENABLE_PROMETHEUS:
    try:
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        logger.info("Prometheus metrics instrumentation mapped at /metrics")
    except Exception as e:
        logger.error(f"Failed to initialize Prometheus instrumentation: {e}")


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "api_docs": "/docs",
        "health": f"{settings.API_PREFIX}/health"
    }
