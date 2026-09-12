"""Conversation routes (ADR-0029 §6). Every handler resolves the conversation through the
owner-scoped query in conversations.service — a foreign id is a 404, never a 403 oracle."""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from safety_assistant.api.dependencies.auth import Principal, require_user
from safety_assistant.api.middleware.ratelimit import rate_limited
from safety_assistant.api.routes import query
from safety_assistant.api.routes.query import MAX_QUERY_CHARS
from safety_assistant.conversations import service as convs
from safety_assistant.conversations.service import ScopeNotAuthorized, SourceScope
from safety_assistant.persistence import get_session
from safety_assistant.persistence.models import Conversation, Message, MessageCitation
from safety_assistant.retrieval import ScopeFilter

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=convs.MAX_TITLE)
    workspace_id: uuid.UUID | None = None
    source_scope: SourceScope = Field(default_factory=SourceScope)


class ConversationPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=convs.MAX_TITLE)
    archived: bool | None = None
    source_scope: SourceScope | None = None


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    as_of: datetime.date | None = None
    regulation_keys: list[str] = Field(default_factory=list, max_length=10)
    k: int | None = Field(default=None, ge=1, le=20)


def _conv_view(c: Conversation) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "title": c.title,
        "title_locked": c.title_locked,
        "workspace_id": str(c.workspace_id) if c.workspace_id else None,
        "source_scope": c.source_scope,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
        "archived_at": c.archived_at,
    }


def _citation_view(c: MessageCitation) -> dict[str, Any]:
    return {
        "order": c.citation_order,
        "label": c.citation_label,
        "chunk_id": str(c.chunk_id) if c.chunk_id else None,
        "version_id": str(c.version_id) if c.version_id else None,
        "regulation_key": c.regulation_key,
        "version_label": c.version_label,
        "section_path": c.section_path,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "source_sha256": c.source_sha256,
        "quote_excerpt": c.quote_excerpt,
        # NULL chunk_id = the cited version was re-ingested/removed since the answer was given.
        "evidence_available": c.chunk_id is not None,
    }


def _message_view(m: Message, citations: list[MessageCitation]) -> dict[str, Any]:
    return {
        "id": str(m.id),
        "role": m.role,
        "content": m.content,
        "answer_mode": m.answer_mode,
        "trace_id": m.trace_id,
        "warnings": m.warnings or [],
        "created_at": m.created_at,
        "citations": [_citation_view(c) for c in citations],
    }


def _load(session: Session, principal: Principal, conversation_id: uuid.UUID) -> Conversation:
    conv = convs.get(session, principal, conversation_id)
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return conv


@router.post("", status_code=201, dependencies=[Depends(rate_limited)])
def create_conversation(
    req: ConversationCreate,
    principal: Principal = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    try:
        conv = convs.create(
            session, principal, title=req.title, source_scope=req.source_scope, workspace_id=req.workspace_id
        )
    except ScopeNotAuthorized as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    session.commit()
    return _conv_view(conv)


@router.get("")
def list_conversations(
    q: str | None = Query(default=None, max_length=200),
    archived: bool = False,
    principal: Principal = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    return {"items": [_conv_view(c) for c in convs.list_for(session, principal, q=q, archived=archived)]}


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: uuid.UUID,
    principal: Principal = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    conv = _load(session, principal, conversation_id)
    msgs = convs.messages_for(session, conv)
    cites = convs.citations_for(session, [m.id for m in msgs])
    return {**_conv_view(conv), "messages": [_message_view(m, cites.get(m.id, [])) for m in msgs]}


@router.patch("/{conversation_id}")
def patch_conversation(
    conversation_id: uuid.UUID,
    patch: ConversationPatch,
    principal: Principal = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    conv = _load(session, principal, conversation_id)
    try:
        convs.update(
            session, conv, principal, title=patch.title, archived=patch.archived, source_scope=patch.source_scope
        )
    except ScopeNotAuthorized as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    session.commit()
    return _conv_view(conv)


@router.post("/{conversation_id}/messages", status_code=201, dependencies=[Depends(rate_limited)])
async def post_message(
    conversation_id: uuid.UUID,
    req: MessageCreate,
    request: Request,
    principal: Principal = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Runs the grounded /ask pipeline under the conversation's source scope and persists the exchange."""
    conv = _load(session, principal, conversation_id)
    if conv.archived_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "conversation is archived")
    if "chat:query" not in principal.scopes:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "scope 'chat:query' required")
    # ponytail: document/workspace/private predicate lands with migration 0004 (Phase D);
    # until then the persisted source_scope is recorded and the corpus scope is data-class based.
    scope = ScopeFilter(
        as_of=req.as_of, regulation_keys=tuple(req.regulation_keys), data_classes=principal.data_classes
    )
    context = convs.context_for(session, conv)
    answer = await run_in_threadpool(
        query._answers().answer,  # via module: tests swap the service
        session,
        req.content,
        scope=scope,
        principal=principal.subject,
        scopes=sorted(principal.scopes),
        k=req.k,
        conversation_context=context,
    )
    user_msg, assistant = convs.record_exchange(session, conv, req.content, answer)
    session.commit()
    request.state.trace_id = answer.trace_id
    cites = convs.citations_for(session, [assistant.id])
    return {
        "conversation": _conv_view(conv),
        "user_message": _message_view(user_msg, []),
        "assistant_message": _message_view(assistant, cites.get(assistant.id, [])),
        "answer": answer,
    }
