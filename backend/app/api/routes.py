"""FastAPI routes for RAG chat and health."""

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from backend.app.core.services import get_workflow
from backend.app.metrics import prometheus as prom

router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    query: str
    answer: str
    documents: list[dict[str, Any]] = []
    guardrails: dict[str, Any] = {}
    timing: dict[str, Any] = {}
    warnings: list[str] = []


@router.get("/health")
async def health() -> dict:
    import os

    from backend.app.core.services import get_retriever

    r = get_retriever()
    return {
        "status": "ok",
        "service": "autosafety-rag-backend",
        "chunks": len(r.chunks),
        "embeddings": len(r.embeddings),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "reranker_enabled": os.getenv("ENABLE_RERANKER", "false").lower() == "true",
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    prom.ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    status = "success"
    try:
        logger.info(f"POST /chat query={req.query[:80]}...")
        result = await asyncio.to_thread(get_workflow().run, req.query)
        if result.get("error"):
            status = "error"
            prom.ERRORS_TOTAL.labels(error_type="workflow").inc()

        gr_out = result.get("guardrails", {}).get("output", {})
        warnings = gr_out.get("warnings", [])
        gr_in = result.get("guardrails", {}).get("input", {})
        if gr_in.get("blocked"):
            prom.GUARDRAIL_BLOCKS.labels(
                reason=gr_in.get("block_reason", "unknown")
            ).inc()

        prompt = result.get("prompt", "")
        answer = result.get("answer", "")
        prom.record_llm_usage(prompt, answer, (time.perf_counter() - t0))
        prom.RETRIEVAL_LATENCY.observe(
            (result.get("timing", {}).get("retrieval_ms", 0) or 0) / 1000
        )
        prom.REQUEST_LATENCY.labels(endpoint="chat", status=status).observe(
            time.perf_counter() - t0
        )

        return ChatResponse(
            query=result["query"],
            answer=result["answer"],
            documents=[
                {
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "heading_path": d.get("heading_path"),
                    "regulation": d.get("regulation"),
                    "source": d.get("source"),
                    "rerank_score": d.get("rerank_score"),
                }
                for d in result.get("documents", [])
            ],
            guardrails=result.get("guardrails", {}),
            timing=result.get("timing", {}),
            warnings=warnings,
        )
    except Exception as exc:
        status = "error"
        prom.ERRORS_TOTAL.labels(error_type=type(exc).__name__).inc()
        prom.REQUEST_LATENCY.labels(endpoint="chat", status=status).observe(
            time.perf_counter() - t0
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        prom.ACTIVE_REQUESTS.dec()
