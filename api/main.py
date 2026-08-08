"""Passive-safety RAG FastAPI application."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_agent import router as agent_router
from api.routes_chat import router as chat_router
from api.routes_metrics import router as metrics_router
from api.routes_sources import router as sources_router
from api.routes_upload import router as upload_router
from api.jobs import init_db as init_jobs_db
from api.conversations import init_db as init_conversations_db
from cache.response_cache import init_db as init_response_cache_db
from ingestion.hf_auth import ensure_hf_auth

load_dotenv()
ensure_hf_auth()
init_jobs_db()
init_conversations_db()
init_response_cache_db()

app = FastAPI(
    title="Passive Safety RAG",
    version="0.5.0",
    description="Grounded UNECE regulation Q&A with citable PDF highlights + multi-step agent",
)

origins = [
    o.strip()
    for o in (os.getenv("CORS_ORIGINS") or "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, tags=["chat"])
app.include_router(agent_router, tags=["agent"])
app.include_router(sources_router, tags=["sources"])
app.include_router(upload_router, tags=["upload"])
app.include_router(metrics_router, tags=["metrics"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "passive-safety-rag"}


def main() -> None:
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
