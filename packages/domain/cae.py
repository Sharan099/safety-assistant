"""CAE (LS-DYNA) structured entities — TRD_LEVEL3.md §19.

`cae_decks` / `cae_files` / `cae_keywords` / `cae_includes` are always
populated by `packages/cae/lsdyna/persistence.py` for every parsed deck.
`cae_parts` / `cae_materials` / `cae_sections` / `cae_contacts` /
`cae_controls` / `cae_databases` are populated only for the `structured:
true` keyword roots (`packages/cae/lsdyna/keyword_registry.yaml`) — there is
deliberately no `cae_nodes`/`cae_elements` table (TRD_LEVEL3.md §19's own
schema list has none; bulk nodal/element geometry belongs in Parquet, not
relational rows, if it's ever needed at all).

`deck_id` is denormalized onto every entity table (in addition to the more
precise `file_id`) because structured search (packages/retrieval/structured.py)
almost always needs to scope a query to "within this one deck" — part/
material/section ids are only unique inside a deck's own numbering, not
globally — and that would otherwise mean a join through `cae_files` on
every single query.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from packages.domain.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class CaeDeck(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "cae_decks"

    knowledge_source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_sources.id"), index=True
    )
    deck_key: Mapped[str] = mapped_column(Text, unique=True)  # e.g. "<source_id>::<main file relpath>"
    main_file_relpath: Mapped[str] = mapped_column(Text)
    # COMPLETE | INCOMPLETE — packages/cae/lsdyna/include_graph.py's per-deck status
    include_status: Mapped[str] = mapped_column(Text)
    parser_version: Mapped[str] = mapped_column(Text)


class CaeFile(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "cae_files"

    deck_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_decks.id"), index=True)
    relpath: Mapped[str] = mapped_column(Text)  # resolution key used by include_graph — basename-unique per deck
    archive_member_path: Mapped[str | None] = mapped_column(Text)  # None for a standalone (non-archived) file
    sha256: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer)
    is_main: Mapped[bool] = mapped_column(Boolean, default=False)


class CaeKeyword(Base, UUIDPrimaryKeyMixin):
    """One row per `RawCard` — every keyword occurrence, structured or not.
    This is the generic, never-loses-anything layer TRD_LEVEL3.md §19 calls
    for; the six *_parts/materials/... tables below are additive detail on
    top, not a replacement."""

    __tablename__ = "cae_keywords"

    deck_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_decks.id"), index=True)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_files.id"), index=True)
    keyword: Mapped[str] = mapped_column(Text, index=True)
    root: Mapped[str | None] = mapped_column(Text)  # matched registry root, or NULL if genuinely unrecognized
    line_start: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    raw_hash: Mapped[str] = mapped_column(Text)


class CaeInclude(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "cae_includes"

    deck_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_decks.id"), index=True)
    parent_file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_files.id"), index=True)
    target_as_written: Mapped[str] = mapped_column(Text)
    resolved_file_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_files.id"))
    # RESOLVED | MISSING | CYCLE | DUPLICATE | AMBIGUOUS | OUTSIDE_ROOT
    status: Mapped[str] = mapped_column(Text)


class CaePart(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "cae_parts"

    deck_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_decks.id"), index=True)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_files.id"), index=True)
    part_id: Mapped[int | None] = mapped_column(Integer, index=True)
    title: Mapped[str | None] = mapped_column(Text)
    section_id: Mapped[int | None] = mapped_column(Integer)
    material_id: Mapped[int | None] = mapped_column(Integer)
    parse_status: Mapped[str] = mapped_column(Text)  # PARSED | PARTIAL | UNKNOWN
    line_start: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    raw_hash: Mapped[str] = mapped_column(Text)


class CaeMaterial(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "cae_materials"

    deck_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_decks.id"), index=True)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_files.id"), index=True)
    material_id: Mapped[int | None] = mapped_column(Integer, index=True)
    mat_type: Mapped[str | None] = mapped_column(Text)
    parse_status: Mapped[str] = mapped_column(Text)
    line_start: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    raw_hash: Mapped[str] = mapped_column(Text)


class CaeSection(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "cae_sections"

    deck_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_decks.id"), index=True)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_files.id"), index=True)
    section_id: Mapped[int | None] = mapped_column(Integer, index=True)
    section_type: Mapped[str | None] = mapped_column(Text)
    parse_status: Mapped[str] = mapped_column(Text)
    line_start: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    raw_hash: Mapped[str] = mapped_column(Text)


class CaeContact(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "cae_contacts"

    deck_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_decks.id"), index=True)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_files.id"), index=True)
    contact_type: Mapped[str | None] = mapped_column(Text)
    ssid: Mapped[int | None] = mapped_column(Integer)
    msid: Mapped[int | None] = mapped_column(Integer)
    parse_status: Mapped[str] = mapped_column(Text)
    line_start: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    raw_hash: Mapped[str] = mapped_column(Text)


class CaeControl(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "cae_controls"

    deck_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_decks.id"), index=True)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_files.id"), index=True)
    control_type: Mapped[str | None] = mapped_column(Text)
    parse_status: Mapped[str] = mapped_column(Text)
    line_start: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    raw_hash: Mapped[str] = mapped_column(Text)


class CaeDatabase(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "cae_databases"

    deck_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_decks.id"), index=True)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("cae_files.id"), index=True)
    database_type: Mapped[str | None] = mapped_column(Text)
    dt: Mapped[float | None] = mapped_column()
    parse_status: Mapped[str] = mapped_column(Text)
    line_start: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    raw_hash: Mapped[str] = mapped_column(Text)
