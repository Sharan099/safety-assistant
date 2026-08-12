"""FastAPI entrypoint — TRD.md §1 Architecture.

Run: `uv run uvicorn apps.api.main:app --reload`
"""

from __future__ import annotations

from fastapi import FastAPI

from apps.api.routers import investigations, knowledge, runs

app = FastAPI(
    title="Passive Safety CAE Investigation Agent",
    version="0.1.0",
    description="Engineering investigation workstation API — see PRD.md / TRD.md.",
)

app.include_router(runs.router, prefix="/api/v1")
app.include_router(investigations.router, prefix="/api/v1")
app.include_router(knowledge.router, prefix="/api/v1")


@app.get("/api/v1/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
