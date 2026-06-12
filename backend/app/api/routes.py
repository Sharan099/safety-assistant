"""FastAPI routes: chat, health, readiness, users, and feedback."""

import asyncio
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from backend.app.core import store
from backend.app.core.services import get_workflow, readiness
from backend.app.core.security import rate_limit
from backend.app.metrics import prometheus as prom

router = APIRouter()


# ───────────────────────── models ─────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    user_id: str | None = Field(default=None, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    query: str
    answer: str
    message_id: str | None = None
    documents: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    grounding: dict[str, Any] = {}
    guardrails: dict[str, Any] = {}
    gateway: dict[str, Any] = {}
    timing: dict[str, Any] = {}
    warnings: list[str] = []


class UserRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=40)


class UserResponse(BaseModel):
    user_id: str
    username: str
    created: bool
    session_id: str


class FeedbackRequest(BaseModel):
    rating: str = Field(..., pattern="^(up|down)$")
    message_id: str | None = Field(default=None, max_length=64)
    user_id: str | None = Field(default=None, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)
    reasons: list[str] = Field(default_factory=list, max_length=10)
    comment: str | None = Field(default=None, max_length=4000)
    query: str | None = Field(default=None, max_length=4000)
    answer: str | None = Field(default=None, max_length=8000)


# ───────────────────────── health / readiness ─────────────────────────
@router.get("/health")
async def health() -> dict:
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


@router.get("/ready")
async def ready() -> dict:
    """
    Self-test gate. The frontend polls this and only enables chat once
    `ready` is true. The first call runs one end-to-end query and is cached.
    """
    result = await asyncio.to_thread(readiness)
    return result


# ───────────────────────── users ─────────────────────────
@router.post("/users", response_model=UserResponse)
async def register_user(req: UserRequest, request: Request) -> UserResponse:
    rate_limit(request, "users", limit=20, window_s=60)
    try:
        user = await asyncio.to_thread(store.get_or_create_user, req.username)
        session_id = await asyncio.to_thread(store.create_session, user["id"], None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("User registration failed")
        raise HTTPException(status_code=500, detail="Could not register user") from exc
    return UserResponse(
        user_id=user["id"],
        username=user["username"],
        created=user["created"],
        session_id=session_id,
    )


# ───────────────────────── feedback ─────────────────────────
@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request) -> dict:
    rate_limit(request, "feedback", limit=60, window_s=60)
    try:
        fid = await asyncio.to_thread(
            store.record_feedback,
            message_id=req.message_id,
            session_id=req.session_id,
            user_id=req.user_id,
            rating=req.rating,
            reasons=req.reasons,
            comment=req.comment,
            query=req.query,
            answer=req.answer,
        )
        prom.FEEDBACK_TOTAL.labels(rating=req.rating).inc()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Feedback save failed")
        raise HTTPException(status_code=500, detail="Could not save feedback") from exc
    return {"status": "ok", "feedback_id": fid}


# ───────────────────────── chat ─────────────────────────
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    rate_limit(request, "chat", limit=30, window_s=60)
    prom.ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    status = "success"
    try:
        logger.info(f"POST /chat query={req.query[:80]}...")
        result = await asyncio.to_thread(
            get_workflow().run, req.query, req.user_id, req.session_id
        )
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

        # Persist the turn (background) so feedback can reference it.
        message_id = None
        try:
            grounding = result.get("grounding", {})
            gateway_meta = result.get("gateway", {})
            message_id = await asyncio.to_thread(
                store.record_message,
                session_id=req.session_id,
                user_id=req.user_id,
                query=result["query"],
                answer=answer,
                grounding=str(grounding) if grounding else None,
                model=gateway_meta.get("model"),
            )
        except Exception as exc:  # never fail the chat because of logging
            logger.warning(f"Message persistence failed: {exc}")

        # On abstention there is no grounded answer: never surface sources/flags.
        should_abstain = bool(grounding.get("should_abstain"))
        citations = [] if should_abstain else result.get("citations", [])
        flags = [] if should_abstain else result.get("flags", [])

        return ChatResponse(
            query=result["query"],
            answer=result["answer"],
            message_id=message_id,
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
            citations=citations,
            flags=flags,
            grounding=result.get("grounding", {}),
            guardrails=result.get("guardrails", {}),
            gateway=result.get("gateway", {}),
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
