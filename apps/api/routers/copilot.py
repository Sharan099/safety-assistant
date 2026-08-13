"""Copilot API — CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 4/11,
PRD_COPILOT_UPDATE.md Section 10.

`POST .../copilot/messages` streams Server-Sent Events: one "step" event per
LangGraph node as it completes (this is what gives the UI real-time
workflow visibility — packages/agent/copilot.py's `stream_copilot_turn`
wraps LangGraph's own `.stream()`, no separate mechanism), then one "final"
event carrying exactly the shape PRD_COPILOT_UPDATE.md Section 10 specifies:
assistant message, evidence references, tool activity, suggested actions,
unknowns.

The session is opened and closed *inside* the generator, not via FastAPI's
`Depends(get_db)` — a `StreamingResponse`'s generator body runs after the
endpoint function has already returned, by which point a dependency-scoped
session would already be closed.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas import CopilotMessageRequest, CopilotMessageSummary, CopilotToolActivity
from packages.agent.copilot import stream_copilot_turn
from packages.agent.llm import LLMProvider, get_provider
from packages.domain.copilot import CopilotConversation, CopilotMessage, CopilotToolCall
from packages.domain.db import get_engine, get_settings

router = APIRouter(tags=["copilot"])


def _get_llm_or_none() -> LLMProvider | None:
    settings = get_settings()
    if not settings.llm_provider or settings.llm_provider == "none":
        return None
    try:
        return get_provider(settings)
    except ValueError:
        # Unconfigured provider (e.g. LLM_MODEL unset) — proceed
        # deterministic-only, per TRD.md Section 30.
        return None


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/investigations/{investigation_id}/copilot/messages")
def send_copilot_message(investigation_id: uuid.UUID, body: CopilotMessageRequest) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        with Session(get_engine()) as session:
            try:
                llm = _get_llm_or_none()
                for event in stream_copilot_turn(session, investigation_id, body.message, llm=llm):
                    if event.type == "step":
                        yield _sse("step", {"node": event.node, "detail": event.detail})
                    else:  # "final"
                        state = event.state or {}
                        yield _sse(
                            "final",
                            {
                                "message": state.get("response", ""),
                                "intent": state.get("intent"),
                                "evidence_refs": state.get("evidence_refs", []),
                                "tool_activity": state.get("tool_calls", []),
                                "suggested_actions": state.get("suggested_actions", []),
                                "unknowns": state.get("unknowns", []),
                                "llm_degraded": state.get("llm_degraded", False),
                            },
                        )
            except ValueError as exc:
                # Unknown investigation / empty message — PRD_COPILOT_UPDATE.md
                # Section 11-style honest failure, not a fabricated answer.
                yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/investigations/{investigation_id}/copilot/messages", response_model=list[CopilotMessageSummary])
def list_copilot_messages(
    investigation_id: uuid.UUID, session: Session = Depends(get_db)
) -> list[CopilotMessageSummary]:
    conversation = session.query(CopilotConversation).filter_by(investigation_id=investigation_id).one_or_none()
    if conversation is None:
        return []

    messages = (
        session.query(CopilotMessage)
        .filter_by(conversation_id=conversation.id)
        .order_by(CopilotMessage.created_at)
        .all()
    )
    summaries = []
    for m in messages:
        tool_calls = session.query(CopilotToolCall).filter_by(message_id=m.id).all()
        summaries.append(
            CopilotMessageSummary(
                id=m.id,
                role=m.role,
                content=m.content,
                evidence_refs=m.evidence_refs or [],
                suggested_actions=m.suggested_actions or [],
                unknowns=m.unknowns or [],
                llm_degraded=m.llm_degraded,
                tool_activity=[
                    CopilotToolActivity(tool_name=t.tool_name, status=t.status, result_summary=t.result_summary or "")
                    for t in tool_calls
                ],
                created_at=m.created_at,
            )
        )
    return summaries
