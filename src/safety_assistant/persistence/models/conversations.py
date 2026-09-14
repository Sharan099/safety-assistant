"""Durable conversation history (ADR-0029 §3, 0003). History is continuity, never evidence."""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from safety_assistant.persistence.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class Conversation(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user_updated", "user_id", "updated_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(Text)
    # A user-edited title is never overwritten by generated titles (03_UI_UX "Conversation history").
    title_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    # {"scopes": [...], "workspace_ids": [...], "document_ids": [...]} — validated in conversations.service
    source_scope: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # Optional context-compression summary. Conversation-scoped, regenerable, never cited.
    summary: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class Message(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("conversation_id", "ordinal", name="uq_message_ordinal_per_conversation"),)

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    # Position in the thread; the only ordering key (created_at ties within one transaction).
    ordinal: Mapped[int] = mapped_column(Integer)
    # user | assistant
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    # GENERATED | EVIDENCE_ONLY | ABSTAINED (assistant only)
    answer_mode: Mapped[str | None] = mapped_column(Text)
    abstain_reason: Mapped[str | None] = mapped_column(Text)  # generation.schemas.AbstainReason when ABSTAINED
    model: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str | None] = mapped_column(Text)
    # query_traces.trace_id — no FK: traces may be pruned independently of history.
    trace_id: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list[str] | None] = mapped_column(JSONB)


class MessageCitation(Base, UUIDPrimaryKeyMixin):
    """Citation as rendered at answer time. Label/scope are denormalised so a re-ingested
    (cascade-deleted) chunk leaves the record intact with `chunk_id IS NULL`."""

    __tablename__ = "message_citations"

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL"))
    version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulation_versions.id", ondelete="SET NULL")
    )
    citation_order: Mapped[int] = mapped_column(Integer)
    citation_label: Mapped[str] = mapped_column(Text)
    regulation_key: Mapped[str] = mapped_column(Text)
    version_label: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    source_sha256: Mapped[str] = mapped_column(Text)
    quote_excerpt: Mapped[str | None] = mapped_column(Text)
    retrieval_rank: Mapped[int | None] = mapped_column(Integer)
