"""Provenance entities — BACKEND_SCHEMA.md §5, §46-48.

ProcessingJob, ProcessingStep, SignalTransformation.

`ArtifactHash` from BACKEND_SCHEMA.md §5 is intentionally not modeled as a
separate table: `Artifact.sha256` (packages/domain/core.py) already covers
the one field the schema doc lists for it, and it gives no further column
definition. Revisit only if a concrete need for hash *history* (as opposed
to the current hash) shows up — see docs/ADR/0005.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import ARRAY, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from packages.domain.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class ProcessingJob(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "processing_jobs"

    job_type: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="PENDING")

    source_artifact_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("artifacts.id"))

    started_at: Mapped[datetime.datetime | None] = mapped_column()
    completed_at: Mapped[datetime.datetime | None] = mapped_column()
    error: Mapped[str | None] = mapped_column(Text)

    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class ProcessingStep(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "processing_steps"

    processing_job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("processing_jobs.id"), index=True
    )

    step_name: Mapped[str] = mapped_column(Text)
    software: Mapped[str | None] = mapped_column(Text)
    software_version: Mapped[str | None] = mapped_column(Text)

    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    input_hashes: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    output_hashes: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    status: Mapped[str] = mapped_column(Text, default="PENDING")
    started_at: Mapped[datetime.datetime | None] = mapped_column()
    completed_at: Mapped[datetime.datetime | None] = mapped_column()


class SignalTransformation(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "signal_transformations"

    signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("signals.id"), index=True)

    transformation_type: Mapped[str] = mapped_column(Text)
    algorithm_version: Mapped[str] = mapped_column(Text)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    input_signal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("signals.id"))
