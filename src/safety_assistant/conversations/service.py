"""Conversation persistence (ADR-0029 §3/§7). Every query is scoped by the owning user in SQL.

History is continuity: it is rendered to the model as `<conversation_context>` (wording only)
and is never a citation source — citations can only reference evidence retrieved in the
current request (generation/citations.py).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from safety_assistant.api.dependencies.auth import Principal
from safety_assistant.generation.schemas import AnswerResponse
from safety_assistant.persistence.models import Conversation, Message, MessageCitation

SourceScopeName = Literal["AUTHORITATIVE_ORG", "WORKSPACE", "PRIVATE_USER"]

MAX_TITLE = 200
PRIOR_TURNS = 6  # user+assistant messages rendered as wording context
PRIOR_TURN_CHARS = 1200


class SourceScope(BaseModel):
    """What the user *chose* to search (FR-CHAT-04). Validated against membership; never widened."""

    scopes: list[SourceScopeName] = Field(default=["AUTHORITATIVE_ORG"], min_length=1, max_length=3)
    workspace_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)


class ScopeNotAuthorized(PermissionError):
    pass


def validate_source_scope(scope: SourceScope, principal: Principal) -> SourceScope:
    """Requested workspaces must be ones the principal belongs to; nothing is silently dropped."""
    if any(w not in principal.workspace_ids for w in scope.workspace_ids):
        raise ScopeNotAuthorized("workspace not in the caller's memberships")
    return scope  # bare WORKSPACE = "all my workspaces" (possibly none) — nothing to widen or narrow


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _owned(principal: Principal) -> Select[tuple[Conversation]]:
    return select(Conversation).where(Conversation.user_id == principal.user_id)


def create(
    session: Session,
    principal: Principal,
    *,
    title: str | None,
    source_scope: SourceScope,
    workspace_id: uuid.UUID | None,
) -> Conversation:
    assert principal.user_id is not None
    if workspace_id is not None and workspace_id not in principal.workspace_ids:
        raise ScopeNotAuthorized("workspace not in the caller's memberships")
    conv = Conversation(
        user_id=principal.user_id,
        organization_id=principal.organization_ids[0],
        workspace_id=workspace_id,
        title=(title or "New investigation")[:MAX_TITLE],
        title_locked=bool(title),
        source_scope=validate_source_scope(source_scope, principal).model_dump(mode="json"),
        updated_at=_now(),
    )
    session.add(conv)
    session.flush()
    return conv


def list_for(
    session: Session, principal: Principal, *, q: str | None = None, archived: bool = False, limit: int = 50
) -> list[Conversation]:
    stmt = _owned(principal).order_by(Conversation.updated_at.desc()).limit(limit)
    stmt = stmt.where(Conversation.archived_at.is_not(None) if archived else Conversation.archived_at.is_(None))
    if q:
        stmt = stmt.where(func.lower(Conversation.title).contains(q.lower()))
    return list(session.scalars(stmt).all())


def get(session: Session, principal: Principal, conversation_id: uuid.UUID) -> Conversation | None:
    """Owner-scoped fetch: a foreign id looks exactly like a missing one (04_APP_FLOWS cross-user flow)."""
    return session.scalar(_owned(principal).where(Conversation.id == conversation_id))


def update(
    session: Session,
    conv: Conversation,
    principal: Principal,
    *,
    title: str | None = None,
    archived: bool | None = None,
    source_scope: SourceScope | None = None,
) -> Conversation:
    if title is not None:
        conv.title, conv.title_locked = title[:MAX_TITLE], True
    if archived is not None:
        conv.archived_at = _now() if archived else None
    if source_scope is not None:
        conv.source_scope = validate_source_scope(source_scope, principal).model_dump(mode="json")
    conv.updated_at = _now()
    session.flush()
    return conv


def messages_for(session: Session, conv: Conversation) -> list[Message]:
    return list(
        session.scalars(select(Message).where(Message.conversation_id == conv.id).order_by(Message.ordinal)).all()
    )


def citations_for(session: Session, message_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[MessageCitation]]:
    out: dict[uuid.UUID, list[MessageCitation]] = {}
    if not message_ids:
        return out
    stmt = (
        select(MessageCitation)
        .where(MessageCitation.message_id.in_(message_ids))
        .order_by(MessageCitation.message_id, MessageCitation.citation_order)
    )
    for c in session.scalars(stmt).all():
        out.setdefault(c.message_id, []).append(c)
    return out


def context_for(session: Session, conv: Conversation, *, turns: int = PRIOR_TURNS) -> str | None:
    """Earlier turns as plain text for the model — wording context only, never evidence."""
    recent = messages_for(session, conv)[-turns:]
    if not recent:
        return None
    lines = [conv.summary] if conv.summary else []
    lines += [f"{m.role}: {m.content[:PRIOR_TURN_CHARS]}" for m in recent]
    return "\n".join(lines)


def record_exchange(
    session: Session, conv: Conversation, question: str, answer: AnswerResponse
) -> tuple[Message, Message]:
    """Persist the user turn, the assistant turn and its citations in one unit of work."""
    last = session.scalar(select(func.max(Message.ordinal)).where(Message.conversation_id == conv.id))
    next_ordinal = (last if last is not None else -1) + 1
    user_msg = Message(conversation_id=conv.id, ordinal=next_ordinal, role="user", content=question)
    session.add(user_msg)
    session.flush()
    assistant = Message(
        conversation_id=conv.id,
        ordinal=next_ordinal + 1,
        role="assistant",
        content=answer.answer or "",
        answer_mode=answer.mode,
        abstain_reason=answer.abstain_reason,
        model=str(answer.versions.get("model") or "") or None,
        provider=str(answer.versions.get("provider") or "") or None,
        trace_id=answer.trace_id,
        warnings=answer.warnings or None,
    )
    session.add(assistant)
    session.flush()
    excerpts = {str(e.get("evidence_id")): e for e in answer.evidence}
    for i, c in enumerate(answer.citations):
        ev = excerpts.get(c.evidence_id, {})
        chunk_id = ev.get("chunk_id")
        version_id = ev.get("version_id")
        session.add(
            MessageCitation(
                message_id=assistant.id,
                chunk_id=uuid.UUID(str(chunk_id)) if chunk_id else None,
                version_id=uuid.UUID(str(version_id)) if version_id else None,
                citation_order=i,
                citation_label=c.label,
                regulation_key=c.regulation_key,
                version_label=c.version_label,
                section_path=c.section_path,
                page_start=c.page_start,
                page_end=c.page_end,
                source_sha256=c.source_sha256,
                quote_excerpt=(str(ev.get("content") or "")[:600] or None),
                retrieval_rank=i + 1,
            )
        )
    if not conv.title_locked and conv.title == "New investigation":
        conv.title = question.strip().splitlines()[0][:MAX_TITLE]
    conv.updated_at = _now()
    session.flush()
    return user_msg, assistant
