"""FastAPI entrypoint — TRD.md §1 Architecture.

Run: `uv run uvicorn apps.api.main:app --reload`
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import copilot, investigations, knowledge, runs
from packages.domain.db import get_settings

app = FastAPI(
    title="Passive Safety CAE Investigation Agent",
    version="0.1.0",
    description="Engineering investigation workstation API — see PRD.md / TRD.md.",
)

# Without this, every fetch() from apps/web (a different origin — port 3010
# vs the API's 8010) is blocked by the browser before it even reaches a
# route, surfacing as an opaque "Failed to fetch" in the frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router, prefix="/api/v1")
app.include_router(investigations.router, prefix="/api/v1")
app.include_router(knowledge.router, prefix="/api/v1")
app.include_router(copilot.router, prefix="/api/v1")


@app.get("/api/v1/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
