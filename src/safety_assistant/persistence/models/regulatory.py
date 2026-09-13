"""Canonical regulatory model — CLAUDE.md §5.1.

Traceability chain every answer must satisfy:

    answer → evidence id → chunk → section → regulation_version
           → source_artifact (sha256, source_uri, dates) → parser/chunker/index versions

Identity vs. version vs. artifact are deliberately separate tables:

- ``regulations``          what the document *is* (UN R94), stable across amendments;
- ``regulation_versions``  one consolidated text with its own validity window and
                           lifecycle state; only ``ACTIVE`` versions are retrievable;
- ``source_artifacts``     the immutable bytes a version was parsed from.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from safety_assistant.config.settings import EMBEDDING_DIMENSIONS
from safety_assistant.persistence.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class Regulation(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "regulations"
    __table_args__ = (
        CheckConstraint("scope <> 'WORKSPACE' OR workspace_id IS NOT NULL", name="ck_regulations_workspace_scope"),
        CheckConstraint("scope <> 'PRIVATE_USER' OR owner_user_id IS NOT NULL", name="ck_regulations_private_scope"),
        Index("ix_regulations_org_scope", "organization_id", "scope"),
        Index("ix_regulations_workspace_id", "workspace_id"),
        Index("ix_regulations_owner_user_id", "owner_user_id"),
    )

    # Stable human key, e.g. "UN-R94", "NHTSA-THOR-05F-QUAL". Unique.
    regulation_key: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    # REGULATION | STANDARD | TECHNICAL_REPORT | MANUAL — supporting official
    # documents live in the same corpus but never masquerade as regulations.
    kind: Mapped[str] = mapped_column(Text)
    # Issuing body and legal scope, e.g. authority="UNECE", jurisdiction="UNECE-1958-AGREEMENT".
    authority: Mapped[str] = mapped_column(Text)
    jurisdiction: Mapped[str] = mapped_column(Text)
    # AUTHORITATIVE | OFFICIAL_DOCUMENTATION | INTERNAL_APPROVED | HISTORICAL | REFERENCE | SYNTHETIC
    authority_level: Mapped[str] = mapped_column(Text)
    # Data classification for provider policy (M11): PUBLIC | CONFIDENTIAL.
    data_class: Mapped[str] = mapped_column(Text, default="PUBLIC")
    # Source scope + ownership (ADR-0029 §1/§4). A `regulations` row is the logical *Document*:
    # AUTHORITATIVE_ORG (registry / promoted) | WORKSPACE (workspace_id) | PRIVATE_USER (owner_user_id).
    # The authorization predicate (retrieval/authz.py) is evaluated on these columns before ranking.
    scope: Mapped[str] = mapped_column(Text, default="AUTHORITATIVE_ORG")
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("workspaces.id"))
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    # Archived documents leave every retrieval set (current and historical) but keep their rows.
    archived_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class SourceArtifact(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """Immutable, content-addressed source bytes. Never updated in place."""

    __tablename__ = "source_artifacts"

    sha256: Mapped[str] = mapped_column(Text, unique=True)
    storage_uri: Mapped[str] = mapped_column(Text)  # object-store key, server-generated
    filename: Mapped[str] = mapped_column(Text)
    media_type: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer)

    source_key: Mapped[str] = mapped_column(Text, index=True)  # registry id (allowlist)
    source_uri: Mapped[str | None] = mapped_column(Text)  # official URI
    etag: Mapped[str | None] = mapped_column(Text)
    last_modified: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class RegulationVersion(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "regulation_versions"
    __table_args__ = (
        UniqueConstraint("regulation_id", "version_label", name="uq_regulation_version_label"),
        Index("ix_regulation_versions_status_valid", "status", "valid_from", "valid_to"),
    )

    regulation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("regulations.id"), index=True)
    source_artifact_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("source_artifacts.id"))

    # e.g. "Rev.4 (04 series)" — the consolidated text's own label; never the parser version.
    version_label: Mapped[str] = mapped_column(Text)
    # Amendment series / revision components when known, e.g. series="04", revision="Rev.4".
    series: Mapped[str | None] = mapped_column(Text)
    revision: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str] = mapped_column(Text, default="en")

    published_at: Mapped[datetime.date | None] = mapped_column(Date)
    valid_from: Mapped[datetime.date | None] = mapped_column(Date)
    valid_to: Mapped[datetime.date | None] = mapped_column(Date)
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("regulation_versions.id"))
    superseded_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    # Lifecycle (safety_assistant.domain.regulations.lifecycle.VersionStatus):
    # DISCOVERED → DOWNLOADED → VALIDATED → PARSED → NORMALIZED → CHUNKED → INDEXED → VERIFIED → ACTIVE
    # plus terminal SUPERSEDED / QUARANTINED / FAILED.
    status: Mapped[str] = mapped_column(Text, default="DISCOVERED", index=True)
    activated_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    # Idempotency / lineage keys (CLAUDE.md §6).
    parser_name: Mapped[str | None] = mapped_column(Text)
    parser_version: Mapped[str | None] = mapped_column(Text)
    parser_config_hash: Mapped[str | None] = mapped_column(Text)
    parsed_hash: Mapped[str | None] = mapped_column(Text)  # sha256 of normalized structure
    chunker_version: Mapped[str | None] = mapped_column(Text)
    chunker_config_hash: Mapped[str | None] = mapped_column(Text)
    index_schema_version: Mapped[int | None] = mapped_column(Integer)

    # Amendment chain parsed from the cover page, e.g.
    # [{"label": "04 series of amendments", "entry_into_force": "2021-06-09"}, ...]
    amendments: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    extraction_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class Section(Base, UUIDPrimaryKeyMixin):
    """A structural node: body clause, annex, annex clause, definition, front matter."""

    __tablename__ = "sections"
    __table_args__ = (
        UniqueConstraint("version_id", "path", name="uq_section_path_per_version"),
        Index("ix_sections_version_ordinal", "version_id", "ordinal"),
    )

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulation_versions.id", ondelete="CASCADE"), index=True
    )
    parent_section_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("sections.id"))

    ordinal: Mapped[int] = mapped_column(Integer)  # document order
    # Materialized path, e.g. "5.2.1.8" or "annex-3/1.4.3.5.2.1" or "front-matter".
    path: Mapped[str] = mapped_column(Text)
    section_number: Mapped[str | None] = mapped_column(Text)  # "5.2.1.8."
    annex: Mapped[str | None] = mapped_column(Text)  # "Annex 3" when inside an annex
    title: Mapped[str | None] = mapped_column(Text)
    # CLAUSE | ANNEX | DEFINITION | FRONT_MATTER | TABLE_OF_CONTENTS
    kind: Mapped[str] = mapped_column(Text)
    # True for requirement text, False for informative material, NULL when unknown.
    normative: Mapped[bool | None] = mapped_column(Boolean)
    depth: Mapped[int] = mapped_column(Integer)

    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    content_sha256: Mapped[str] = mapped_column(Text, index=True)


class Table(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "tables"

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulation_versions.id", ondelete="CASCADE"), index=True
    )
    section_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("sections.id"))
    page_number: Mapped[int] = mapped_column(Integer)
    table_index: Mapped[int] = mapped_column(Integer)
    caption: Mapped[str | None] = mapped_column(Text)
    headers: Mapped[list[str | None] | None] = mapped_column(JSONB)
    rows: Mapped[list[list[str | None]]] = mapped_column(JSONB)
    bounding_box: Mapped[dict[str, float] | None] = mapped_column(JSONB)
    extraction_method: Mapped[str] = mapped_column(Text)
    quality_score: Mapped[float] = mapped_column(Double)
    content_sha256: Mapped[str] = mapped_column(Text)


class Figure(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "figures"

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulation_versions.id", ondelete="CASCADE"), index=True
    )
    section_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("sections.id"))
    page_number: Mapped[int] = mapped_column(Integer)
    figure_index: Mapped[int] = mapped_column(Integer)
    caption: Mapped[str | None] = mapped_column(Text)
    storage_uri: Mapped[str] = mapped_column(Text)
    image_sha256: Mapped[str] = mapped_column(Text)
    bounding_box: Mapped[dict[str, float] | None] = mapped_column(JSONB)
    figure_type: Mapped[str | None] = mapped_column(Text)


class CrossReference(Base, UUIDPrimaryKeyMixin):
    """A textual reference from one section to another ("Annex 3, paragraph 1.4.3.")."""

    __tablename__ = "cross_references"

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulation_versions.id", ondelete="CASCADE"), index=True
    )
    from_section_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sections.id"), index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    target_path: Mapped[str] = mapped_column(Text)  # normalized path the text points at
    target_regulation_key: Mapped[str | None] = mapped_column(Text)  # cross-regulation refs
    resolved_section_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("sections.id"))


class Chunk(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """Retrieval unit. ``id`` is deterministic: uuid5(version_id, ordinal, chunk_sha256)."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("version_id", "ordinal", name="uq_chunk_ordinal_per_version"),
        Index("ix_chunks_version_sha", "version_id", "chunk_sha256"),
    )

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulation_versions.id", ondelete="CASCADE"), index=True
    )
    section_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sections.id"), index=True)
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("chunks.id"))

    ordinal: Mapped[int] = mapped_column(Integer)
    # TEXT | TABLE | DEFINITION | PARENT
    chunk_type: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    # Exactly what a citation renders: "UN R94 Rev.4 §5.2.1.8 (p. 13)".
    citation_label: Mapped[str] = mapped_column(Text)
    chunk_sha256: Mapped[str] = mapped_column(Text)
    # Summary-augmented retrieval representation (contextualization/): document identity block +
    # document summary + `content`. Indexed by the "sac" representation only; NEVER served as
    # evidence — `content` is the only text an answer may quote. NULL until contextualized.
    retrieval_text: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class DocumentSummary(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """One generated retrieval summary per document version, cached by
    (artifact sha256, prompt version, model). Retrieval metadata, never evidence."""

    __tablename__ = "document_summaries"
    __table_args__ = (UniqueConstraint("version_id", "cache_key", name="uq_document_summary_cache_key"),)

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulation_versions.id", ondelete="CASCADE"), index=True
    )
    cache_key: Mapped[str] = mapped_column(Text)
    content_sha256: Mapped[str] = mapped_column(Text)  # the source artifact the summary describes
    prompt_version: Mapped[str] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)  # READY | FAILED
    summary: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class ChunkEmbedding(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id", "model_name", "model_version", "representation", name="uq_embedding_per_chunk_model"
        ),
        # Partial HNSW cosine indexes per representation are created in the migration (raw DDL options).
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(Text)
    # Which text was embedded: "content" (baseline) or "sac_v1" (chunks.retrieval_text). Separate
    # index versions coexist so the SAC index is built and evaluated without touching the baseline.
    representation: Mapped[str] = mapped_column(Text, default="content", server_default="content")
    dimensions: Mapped[int] = mapped_column(Integer)
    # Fixed dimension so an HNSW index can exist (docs/ADR/0019).
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
