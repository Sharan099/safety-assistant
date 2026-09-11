"""Operational/audit tables: ingestion runs + events, query traces,
evaluation cases, user feedback — CLAUDE.md §5.1, §15."""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from safety_assistant.persistence.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class IngestionRun(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "ingestion_runs"

    source_key: Mapped[str] = mapped_column(Text, index=True)
    version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("regulation_versions.id"))
    # RUNNING | SUCCEEDED | FAILED | QUARANTINED | SKIPPED_UNCHANGED
    status: Mapped[str] = mapped_column(Text, default="RUNNING", index=True)
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    error: Mapped[str | None] = mapped_column(Text)
    git_sha: Mapped[str | None] = mapped_column(Text)
    # counts per stage, durations, reused/recomputed flags, freshness lag seconds
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class IngestionEvent(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "ingestion_events"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ingestion_runs.id", ondelete="CASCADE"), index=True
    )
    at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    from_status: Mapped[str | None] = mapped_column(Text)
    to_status: Mapped[str] = mapped_column(Text)
    level: Mapped[str] = mapped_column(Text, default="INFO")  # INFO | WARNING | ERROR
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class QueryTrace(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """One row per /ask or /search request — the full retrieval/generation record."""

    __tablename__ = "query_traces"

    trace_id: Mapped[str] = mapped_column(Text, unique=True)
    principal: Mapped[str | None] = mapped_column(Text)  # subject id, never a secret
    scopes: Mapped[list[str] | None] = mapped_column(JSONB)
    query: Mapped[str] = mapped_column(Text)
    filters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # intent, rewrites, as_of, route
    candidates: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)  # id, leg ranks, fused, rerank
    evidence: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)  # selected evidence ids + labels
    answer: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # structured output or abstention
    validation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # citation validation outcome
    versions: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # model/prompt/index/parser versions
    latency_ms: Mapped[dict[str, float] | None] = mapped_column(JSONB)  # per stage
    tokens: Mapped[dict[str, int] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)


class EvaluationCase(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "evaluation_cases"

    case_id: Mapped[str] = mapped_column(Text, unique=True)
    dataset_version: Mapped[str] = mapped_column(Text, index=True)
    query_type: Mapped[str] = mapped_column(Text, index=True)
    review_status: Mapped[str] = mapped_column(Text, default="DRAFT")  # DRAFT | REVIEWED | REJECTED
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)  # full gold case (evals/datasets schema)


class UserFeedback(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "user_feedback"

    trace_id: Mapped[str] = mapped_column(Text, index=True)
    principal: Mapped[str | None] = mapped_column(Text)
    rating: Mapped[int] = mapped_column(Integer)  # -1 | 0 | 1
    comment: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
