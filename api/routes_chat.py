"""POST /chat — SSE streaming grounded answers with structured citations."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError as PydanticValidationError

from api.validation import ChatInput, ValidationError
from generation.answer import AnswerResponse, SourceChunk, answer_question
from generation.llm_client import LLMClient

logger = logging.getLogger(__name__)
router = APIRouter()

CITATION_RE = re.compile(
    r"\[(?P<reg>[^\]]+?)\s*§(?P<section>[^,\]\s]+)\s*,\s*p\.(?P<page>\d+|\?)\]"
)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _citation_payload(sources: list[SourceChunk]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in sources:
        short_reg = (s.regulation_id or "").replace("UN-ECE-", "")
        label = f"[{short_reg or s.regulation_id} §{s.section_number or '?'}, p.{s.page_number if s.page_number is not None else '?'}]"
        out.append(
            {
                "chunk_id": s.chunk_id,
                "regulation_id": s.regulation_id,
                "section_number": s.section_number,
                "section_title": s.section_title,
                "page_number": s.page_number,
                "bounding_box": s.bounding_box,
                "coord_origin": "BOTTOMLEFT",
                "citation": s.citation or label,
                "label": label,
                "text": (s.text or "")[:500],
                "score": s.score,
            }
        )
    return out


def _metrics_summary(result: AnswerResponse) -> dict[str, Any]:
    m = result.metrics or {}
    llm_calls = m.get("llm_calls") or []
    answer_call = next((c for c in reversed(llm_calls) if c.get("role") == "answer"), None)
    return {
        "trace_id": result.trace_id,
        "input_tokens": m.get("input_tokens", 0),
        "output_tokens": m.get("output_tokens", 0),
        "embedding_tokens": m.get("embedding_tokens", 0),
        "rerank_calls": m.get("rerank_calls", 0),
        "cost_usd": m.get("cost_usd", 0.0),
        "latency_ms": m.get("latency_ms", 0.0),
        "model": m.get("model") or result.model,
        "provider": m.get("answer_provider")
        or m.get("provider")
        or result.provider,
        "cache_status": (answer_call or {}).get("cache_status")
        or m.get("cache_hit_kind")
        or "",
        "llm_calls": llm_calls,
        "chunk_ids": m.get("chunk_ids") or [s.chunk_id for s in result.sources],
        "not_found": result.not_found,
        "failure_kind": result.failure_kind,
        "answer_cached": result.answer_cached,
        "served_from_cache": result.served_from_cache,
        "cache_hit_kind": result.cache_hit_kind or m.get("cache_hit_kind") or "",
        "prompt_cache_hit": m.get("prompt_cache_hit", False),
    }


async def _stream_answer(req: ChatInput) -> AsyncIterator[bytes]:
    yield _sse("status", {"stage": "retrieving"}).encode("utf-8")
    await asyncio.sleep(0)

    try:
        result: AnswerResponse = await asyncio.to_thread(
            answer_question,
            req.question,
            top_k=req.top_k,
            regulation_id=req.regulation_id,
            conversation_id=req.conversation_id,
            llm=LLMClient(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed")
        yield _sse("error", {"message": str(exc), "code": "pipeline_error"}).encode("utf-8")
        return

    if result.not_found:
        yield _sse(
            "not_found",
            {
                "message": result.answer,
                "failure_kind": result.failure_kind,
                "trace_id": result.trace_id,
                "conversation_id": result.conversation_id,
                "condensed_question": result.condensed_question,
            },
        ).encode("utf-8")

    citations = _citation_payload(result.sources)
    yield _sse(
        "citations",
        {
            "citations": citations,
            "question": result.question,
            "condensed_question": result.condensed_question,
            "conversation_id": result.conversation_id,
        },
    ).encode("utf-8")

    text = result.answer or ""
    parts = re.findall(r"\S+\s*|\s+", text) or [text]
    buf = ""
    for part in parts:
        buf += part
        if len(buf) >= 12 or part.endswith(("\n", ". ", "? ", "! ")):
            yield _sse("token", {"text": buf}).encode("utf-8")
            buf = ""
            await asyncio.sleep(0.01)
    if buf:
        yield _sse("token", {"text": buf}).encode("utf-8")

    yield _sse(
        "done",
        {
            "model": result.model,
            "provider": result.provider,
            "cached": result.cached,
            "answer_cached": result.answer_cached,
            "served_from_cache": result.served_from_cache,
            "cache_hit_kind": result.cache_hit_kind,
            "not_found": result.not_found,
            "failure_kind": result.failure_kind,
            "answer": result.answer,
            "citations": citations,
            "compliance": result.compliance,
            "query_intent": result.query_intent,
            "mode_disclaimer": result.mode_disclaimer,
            "mode_disclaimer_title": result.mode_disclaimer_title,
            "execution_layer": result.execution_layer,
            "multi_step": result.multi_step,
            "trace_id": result.trace_id,
            "metrics": _metrics_summary(result),
            "conversation_id": result.conversation_id,
            "condensed_question": result.condensed_question,
            "condensation_applied": result.condensation_applied,
        },
    ).encode("utf-8")


@router.post("/chat")
async def chat(body: dict[str, Any]):
    """SSE stream: status → citations → token* → done."""
    try:
        req = ChatInput.model_validate(body)
    except (ValidationError, PydanticValidationError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return StreamingResponse(
        _stream_answer(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/sync")
async def chat_sync(body: dict[str, Any]) -> dict[str, Any]:
    """Non-streaming JSON for debugging / eval."""
    try:
        req = ChatInput.model_validate(body)
    except (ValidationError, PydanticValidationError) as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        result = await asyncio.to_thread(
            answer_question,
            req.question,
            top_k=req.top_k,
            regulation_id=req.regulation_id,
            conversation_id=req.conversation_id,
            llm=LLMClient(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat/sync failed")
        raise HTTPException(500, str(exc)) from exc
    return {
        "question": result.question,
        "answer": result.answer,
        "citations": _citation_payload(result.sources),
        "model": result.model,
        "provider": result.provider,
        "cached": result.cached,
        "answer_cached": result.answer_cached,
        "served_from_cache": result.served_from_cache,
        "cache_hit_kind": result.cache_hit_kind,
        "not_found": result.not_found,
        "failure_kind": result.failure_kind,
        "trace_id": result.trace_id,
        "metrics": _metrics_summary(result),
        "conversation_id": result.conversation_id,
        "condensed_question": result.condensed_question,
        "condensation_applied": result.condensation_applied,
        "compliance": result.compliance,
        "query_intent": result.query_intent,
        "mode_disclaimer": result.mode_disclaimer,
        "mode_disclaimer_title": result.mode_disclaimer_title,
    }
