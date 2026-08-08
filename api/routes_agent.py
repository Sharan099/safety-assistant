"""POST /agent — multi-step grounded agent (compare / report / qa)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from agent.loop import run_agent
from agent.state import AgentResult
from api.validation import ValidationError, validate_question
from generation.llm_client import LLMClient

logger = logging.getLogger(__name__)
router = APIRouter()


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=4000)
    stream: bool = False


def _serialize(result: AgentResult) -> dict[str, Any]:
    return {
        "task": result.task,
        "mode": result.mode,
        "answer": result.answer,
        "table_markdown": result.table_markdown,
        "report_markdown": result.report_markdown,
        "sources": [s.model_dump() for s in result.sources],
        "plan": [p.model_dump() for p in result.plan],
        "steps": [
            {
                "step_id": s.step_id,
                "tool": s.tool,
                "args": s.args,
                "output_preview": s.output_preview,
                "citations": s.citations,
                "chunk_ids": s.chunk_ids,
                "citation_coverage": s.citation_coverage,
                "ungrounded": sum(1 for c in s.claim_checks if not c.grounded),
                "latency_ms": s.latency_ms,
                "error": s.error,
            }
            for s in result.steps
        ],
        "trace_id": result.trace_id,
        "provider": result.provider,
        "model": result.model,
        "not_found": result.not_found,
        "overall_citation_coverage": result.overall_citation_coverage,
        "ungrounded_claim_count": result.ungrounded_claim_count,
        "metrics": result.metrics,
        "query_intent": result.query_intent,
        "execution_layer": result.execution_layer,
        "multi_step": result.multi_step,
        "mode_disclaimer": result.mode_disclaimer,
        "mode_disclaimer_title": result.mode_disclaimer_title,
    }


@router.post("/agent")
async def agent_run(body: dict[str, Any]) -> dict[str, Any]:
    try:
        task = validate_question(str(body.get("task") or body.get("question") or ""))
        req = AgentRequest(task=task, stream=bool(body.get("stream")))
    except (ValidationError, PydanticValidationError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    if req.stream:
        return StreamingResponse(
            _stream(req.task),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await asyncio.to_thread(run_agent, req.task, llm=LLMClient())
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent failed")
        raise HTTPException(500, str(exc)) from exc
    return _serialize(result)


@router.post("/agent/compare")
async def agent_compare(body: dict[str, Any]) -> dict[str, Any]:
    """Convenience: compliance comparison for a topic across two regulations."""
    topic = str(body.get("topic") or "").strip()
    reg_a = str(body.get("reg_a") or body.get("regulation_a") or "R94").strip()
    reg_b = str(body.get("reg_b") or body.get("regulation_b") or "R95").strip()
    if len(topic) < 3:
        raise HTTPException(400, "topic is required")
    task = f"Compare {reg_a} vs {reg_b} {topic}"
    result = await asyncio.to_thread(run_agent, task, llm=LLMClient())
    return _serialize(result)


@router.post("/agent/report")
async def agent_report(body: dict[str, Any]) -> dict[str, Any]:
    """Convenience: draft a cited gap-analysis / engineering memo."""
    title = str(body.get("title") or "Gap analysis memo").strip()
    focus = str(body.get("focus") or body.get("task") or "").strip()
    if len(focus) < 3:
        raise HTTPException(400, "focus/task is required")
    task = f"Gap analysis: {focus}" if "gap" not in focus.lower() else focus
    if title and title.lower() not in task.lower():
        task = f"{title}: {task}"
    result = await asyncio.to_thread(run_agent, task, llm=LLMClient())
    return _serialize(result)


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def _stream(task: str) -> AsyncIterator[bytes]:
    queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_step(step) -> None:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {
                "step_id": step.step_id,
                "tool": step.tool,
                "citation_coverage": step.citation_coverage,
                "citations": step.citations,
                "preview": step.output_preview[:500],
            },
        )

    async def runner() -> None:
        try:
            result = await asyncio.to_thread(
                lambda: run_agent(task, llm=LLMClient(), on_step=on_step)
            )
            await queue.put({"_final": _serialize(result)})
        except Exception as exc:  # noqa: BLE001
            await queue.put({"_error": str(exc)})
        finally:
            await queue.put(None)

    yield _sse("status", {"stage": "planning"})
    task_handle = asyncio.create_task(runner())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if "_error" in item:
                yield _sse("error", {"message": item["_error"]})
                return
            if "_final" in item:
                yield _sse("done", item["_final"])
                return
            yield _sse("step", item)
    finally:
        await task_handle
