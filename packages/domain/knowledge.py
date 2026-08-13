"""Knowledge entities — BACKEND_SCHEMA.md §3, §18-28.

KnowledgeSource, Document, DocumentRevision, SourceSnapshot, DocumentPage,
DocumentSection, DocumentChunk, DocumentTable, DocumentFigure,
DocumentEquation, Embedding.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, Double, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from packages.domain.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


# source_type: AUTHORITATIVE | OFFICIAL_DOCUMENTATION | INTERNAL_APPROVED |
#              HISTORICAL | REFERENCE | SYNTHETIC — PRD.md §11, TRD.md §22
class KnowledgeSource(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "knowledge_sources"

    source_key: Mapped[str] = mapped_column(Text, unique=True)
    source_type: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    authority_level: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    local_path: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class Document(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "documents"

    knowledge_source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_sources.id"), index=True
    )
    document_key: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    document_type: Mapped[str | None] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(Text)


class SourceSnapshot(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "source_snapshots"

    storage_uri: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(Text, unique=True)
    size_bytes: Mapped[int] = mapped_column()

    retrieved_at: Mapped[datetime.datetime | None] = mapped_column()
    source_url: Mapped[str | None] = mapped_column(Text)

    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class DocumentRevision(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "document_revisions"

    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"), index=True)

    revision_label: Mapped[str] = mapped_column(Text)
    effective_date: Mapped[datetime.date | None] = mapped_column(Date)
    source_snapshot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("source_snapshots.id"))

    extractor: Mapped[str | None] = mapped_column(Text)
    extractor_version: Mapped[str | None] = mapped_column(Text)
    # NOT_INGESTED | EXTRACTING | READY | FAILED
    status: Mapped[str] = mapped_column(Text, default="NOT_INGESTED")


class DocumentPage(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "document_pages"

    document_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_revisions.id"), index=True
    )

    page_number: Mapped[int] = mapped_column(Integer)
    text_content: Mapped[str | None] = mapped_column(Text)
    markdown_content: Mapped[str | None] = mapped_column(Text)

    page_image_uri: Mapped[str | None] = mapped_column(Text)

    text_quality: Mapped[float | None] = mapped_column(Double)
    layout_quality: Mapped[float | None] = mapped_column(Double)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_confidence: Mapped[float | None] = mapped_column(Double)

    # DISCOVERED | EXTRACTED | NEEDS_REVIEW | FAILED — PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md
    # §7's page-state list, collapsed to the subset this pipeline can
    # actually populate honestly (no OCR/VLM engine installed — see
    # docs/ADR/0011/0016 — so OCR_REQUIRED/OCR_COMPLETE/VISUAL_REVIEW_REQUIRED
    # would be indistinguishable from NEEDS_REVIEW here; not invented).
    status: Mapped[str] = mapped_column(Text, default="DISCOVERED")

    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class DocumentSection(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "document_sections"

    document_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_revisions.id"), index=True
    )
    parent_section_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_sections.id"))

    title: Mapped[str | None] = mapped_column(Text)
    section_number: Mapped[str | None] = mapped_column(Text)
    start_page: Mapped[int | None] = mapped_column(Integer)
    end_page: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str | None] = mapped_column(Text)


class DocumentChunk(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "document_chunks"

    document_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_revisions.id"), index=True
    )
    section_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_sections.id"))
    page_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_pages.id"))
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_chunks.id"))

    chunk_type: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)

    source_locator: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class Embedding(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "embeddings"

    chunk_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_chunks.id"), index=True)

    model_name: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(Text)
    dimensions: Mapped[int] = mapped_column(Integer)

    # No fixed dimension yet — TRD.md §20: "Do not commit to an embedding
    # model before benchmarking." Once a model is chosen, add a dim and an
    # ivfflat/hnsw index in a follow-up migration.
    embedding: Mapped[Any] = mapped_column(Vector())


class DocumentTable(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "document_tables"

    document_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_revisions.id"), index=True
    )
    page_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_pages.id"))

    table_number: Mapped[str | None] = mapped_column(Text)
    markdown_uri: Mapped[str | None] = mapped_column(Text)
    json_uri: Mapped[str | None] = mapped_column(Text)
    image_uri: Mapped[str | None] = mapped_column(Text)

    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    extraction_method: Mapped[str | None] = mapped_column(Text)
    quality_score: Mapped[float | None] = mapped_column(Double)


class DocumentFigure(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "document_figures"

    document_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_revisions.id"), index=True
    )
    page_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_pages.id"))

    figure_number: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    image_uri: Mapped[str] = mapped_column(Text)
    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    figure_type: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class DocumentEquation(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "document_equations"

    document_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_revisions.id"), index=True
    )
    page_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_pages.id"))

    equation_number: Mapped[str | None] = mapped_column(Text)
    latex: Mapped[str | None] = mapped_column(Text)
    image_uri: Mapped[str | None] = mapped_column(Text)
    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    extraction_method: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Double)
