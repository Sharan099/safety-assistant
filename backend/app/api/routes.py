"""FastAPI routes: chat, health, readiness, users, and feedback."""

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from backend.app.core import store
from backend.app.core.services import get_workflow, readiness
from backend.app.core.security import rate_limit, verify_dashboard_key
from backend.app.metrics import prometheus as prom

router = APIRouter()


# ───────────────────────── models ─────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    user_id: str | None = Field(default=None, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)
    mode: str | None = Field(default=None, max_length=64)
    role: str | None = Field(default="engineer", pattern="^(engineer|manager)$")


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
    generation_failed: bool = False


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
    """Lightweight liveness probe — must not load ML models (Railway healthcheck)."""
    from backend.app.core.services import pipeline_status

    return {
        "status": "ok",
        "service": "autosafety-rag-backend",
        **pipeline_status(),
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
    if not req.user_id:
        raise HTTPException(status_code=422, detail="user_id is required for feedback")
    try:
        if req.session_id:
            await asyncio.to_thread(store.create_session, req.user_id, req.session_id)
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


@router.get("/feedback/dashboard")
async def feedback_dashboard(
    request: Request,
    since: int | None = None,
    user_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """
    Admin dashboard: all users + feedback (poll with ?since=<unix_ts> for updates).
    Requires header X-Dashboard-Key matching FEEDBACK_DASHBOARD_KEY.
    """
    verify_dashboard_key(request)
    rate_limit(request, "feedback_dashboard", limit=120, window_s=60)
    limit = min(max(limit, 1), 500)
    try:
        stats = await asyncio.to_thread(store.feedback_stats)
        users = await asyncio.to_thread(store.list_user_profiles)
        feedback = await asyncio.to_thread(
            store.list_feedback, since=since, user_id=user_id, limit=limit
        )
    except Exception as exc:
        logger.exception("Feedback dashboard load failed")
        raise HTTPException(status_code=500, detail="Could not load dashboard") from exc
    return {
        "server_time": int(time.time()),
        "stats": stats,
        "users": users,
        "feedback": feedback,
    }


# ───────────────────────── chat ─────────────────────────
async def _workflow_result_to_response(
    req: ChatRequest, result: dict[str, Any], t0: float, *, endpoint: str
) -> ChatResponse:
    status = "error" if result.get("error") else "success"
    if result.get("error"):
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
    prom.REQUEST_LATENCY.labels(endpoint=endpoint, status=status).observe(
        time.perf_counter() - t0
    )

    message_id = None
    grounding = result.get("grounding", {})
    try:
        if req.user_id and req.session_id:
            await asyncio.to_thread(store.create_session, req.user_id, req.session_id)
        gateway_meta = result.get("gateway", {})
        from config import GROQ_MODEL

        message_id = await asyncio.to_thread(
            store.record_message,
            session_id=req.session_id,
            user_id=req.user_id,
            query=result["query"],
            answer=answer,
            grounding=str(grounding) if grounding else None,
            model=gateway_meta.get("model") or GROQ_MODEL,
        )
    except Exception as exc:
        logger.warning(f"Message persistence failed: {exc}")

    should_abstain = bool(grounding.get("should_abstain"))
    generation_failed = bool(result.get("generation_failed") or grounding.get("generation_failed"))
    citations = [] if (should_abstain or generation_failed) else result.get("citations", [])
    flags = [] if (should_abstain or generation_failed) else result.get("flags", [])
    if generation_failed:
        grounding = {**grounding, "confidence_band": None, "generation_failed": True}

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
        generation_failed=generation_failed,
    )


@router.get("/modes")
async def list_modes() -> dict:
    from backend.app.core.modes import get_default_mode, list_modes as _list

    return {"default": get_default_mode(), "modes": _list()}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    rate_limit(request, "chat", limit=30, window_s=60)
    prom.ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    try:
        logger.info(f"POST /chat query={req.query[:80]}...")
        result = await asyncio.to_thread(
            get_workflow().run,
            req.query,
            req.user_id,
            req.session_id,
            req.mode,
            req.role,
        )
        return await _workflow_result_to_response(req, result, t0, endpoint="chat")
    except Exception as exc:
        prom.ERRORS_TOTAL.labels(error_type=type(exc).__name__).inc()
        prom.REQUEST_LATENCY.labels(endpoint="chat", status="error").observe(
            time.perf_counter() - t0
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        prom.ACTIVE_REQUESTS.dec()


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """
    NDJSON stream with keepalive pings every ~10s so HF/Vercel gateways do not
    close the connection during slow CPU reranking (60–120s).
    """
    rate_limit(request, "chat", limit=30, window_s=60)
    prom.ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    logger.info(f"POST /chat/stream query={req.query[:80]}...")

    async def ndjson() -> AsyncIterator[str]:
        task = asyncio.create_task(
            asyncio.to_thread(
                get_workflow().run,
                req.query,
                req.user_id,
                req.session_id,
                req.mode,
                req.role,
            )
        )
        try:
            while not task.done():
                yield json.dumps(
                    {"type": "ping", "elapsed_s": round(time.perf_counter() - t0, 1)}
                ) + "\n"
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
                except asyncio.TimeoutError:
                    continue
            result = await task
            response = await _workflow_result_to_response(
                req, result, t0, endpoint="chat_stream"
            )
            yield json.dumps(
                {"type": "result", "data": response.model_dump()}, default=str
            ) + "\n"
        except Exception as exc:
            prom.ERRORS_TOTAL.labels(error_type=type(exc).__name__).inc()
            prom.REQUEST_LATENCY.labels(endpoint="chat_stream", status="error").observe(
                time.perf_counter() - t0
            )
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
        finally:
            prom.ACTIVE_REQUESTS.dec()

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")


# ───────────────────────── documents (upload / manage) ─────────────────────────
class DocumentUploadMeta(BaseModel):
    doc_type: str = Field(default="reference", pattern="^(legal|rating|reference|internal)$")
    authority: str = Field(default="", max_length=120)
    region: str = Field(default="global", max_length=40)
    test_type: str = Field(default="general", max_length=40)
    revision: str = Field(default="", max_length=80)


@router.get("/documents")
async def list_documents() -> dict:
    from backend.app.core.document_service import list_documents

    return {"documents": list_documents()}


@router.get("/documents/{job_id}")
async def document_job_status(job_id: str) -> dict:
    from backend.app.core.document_service import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/documents")
async def upload_document(
    request: Request,
) -> dict:
    from backend.app.core.document_service import start_upload

    form = await request.form()
    upload = form.get("file")
    if upload is None:
        raise HTTPException(status_code=400, detail="file is required")
    file_bytes = await upload.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file")
    meta = DocumentUploadMeta(
        doc_type=str(form.get("doc_type", "reference")),
        authority=str(form.get("authority", "")),
        region=str(form.get("region", "global")),
        test_type=str(form.get("test_type", "general")),
        revision=str(form.get("revision", "")),
    )
    job_id = start_upload(file_bytes, upload.filename, meta.model_dump())
    return {"job_id": job_id, "status": "uploaded"}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    from backend.app.core.document_service import delete_document

    result = delete_document(doc_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "not found"))
    return result
