"""Persists a parsed LS-DYNA deck set + its include graph into the
`cae_*` tables (`packages/domain/cae.py`) — TRD_LEVEL3.md §19.

Idempotent per `deck_key`, same pattern as `packages/ingestion/pipeline.py`'s
`ingest_document`: re-running with the same key returns the existing
`CaeDeck` rather than duplicating rows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from packages.cae.lsdyna.include_graph import IncludeGraph
from packages.cae.lsdyna.models import ParsedDeck
from packages.cae.lsdyna.registry import registry_roots
from packages.cae.lsdyna.registry import root_for as _root_for
from packages.domain.cae import (
    CaeContact,
    CaeControl,
    CaeDatabase,
    CaeDeck,
    CaeFile,
    CaeInclude,
    CaeKeyword,
    CaeMaterial,
    CaePart,
    CaeSection,
)

PARSER_VERSION = "lsdyna-parser v0.1.0"


@dataclass
class FileMeta:
    sha256: str
    size_bytes: int
    archive_member_path: str | None = None


def persist_deck(
    session: Session,
    *,
    knowledge_source_id: uuid.UUID,
    deck_key: str,
    main_file_relpath: str,
    decks: dict[str, ParsedDeck],
    graph: IncludeGraph,
    file_meta: dict[str, FileMeta] | None = None,
    parser_version: str = PARSER_VERSION,
) -> CaeDeck:
    existing = session.query(CaeDeck).filter_by(deck_key=deck_key).one_or_none()
    if existing is not None:
        return existing

    file_meta = file_meta or {}
    roots = registry_roots()

    deck = CaeDeck(
        knowledge_source_id=knowledge_source_id,
        deck_key=deck_key,
        main_file_relpath=main_file_relpath,
        include_status=graph.deck_status.get(main_file_relpath, "INCOMPLETE"),
        parser_version=parser_version,
    )
    session.add(deck)
    session.flush()

    file_rows: dict[str, CaeFile] = {}
    for relpath in decks:
        meta = file_meta.get(relpath)
        row = CaeFile(
            deck_id=deck.id,
            relpath=relpath,
            archive_member_path=meta.archive_member_path if meta else None,
            sha256=meta.sha256 if meta else "",
            size_bytes=meta.size_bytes if meta else 0,
            is_main=(relpath == main_file_relpath),
        )
        session.add(row)
        file_rows[relpath] = row
    session.flush()

    for relpath, parsed in decks.items():
        file_row = file_rows[relpath]
        for card in parsed.all_cards:
            session.add(
                CaeKeyword(
                    deck_id=deck.id,
                    file_id=file_row.id,
                    keyword=card.keyword,
                    root=_root_for(card.keyword, roots),
                    line_start=card.line_start,
                    line_end=card.line_end,
                    raw_hash=card.raw_hash,
                )
            )
        for part in parsed.parts:
            session.add(
                CaePart(
                    deck_id=deck.id,
                    file_id=file_row.id,
                    part_id=part.part_id,
                    title=part.title,
                    section_id=part.section_id,
                    material_id=part.material_id,
                    parse_status=part.parse_status,
                    line_start=part.raw.line_start,
                    line_end=part.raw.line_end,
                    raw_hash=part.raw.raw_hash,
                )
            )
        for section in parsed.sections:
            session.add(
                CaeSection(
                    deck_id=deck.id,
                    file_id=file_row.id,
                    section_id=section.section_id,
                    section_type=section.section_type,
                    parse_status=section.parse_status,
                    line_start=section.raw.line_start,
                    line_end=section.raw.line_end,
                    raw_hash=section.raw.raw_hash,
                )
            )
        for material in parsed.materials:
            session.add(
                CaeMaterial(
                    deck_id=deck.id,
                    file_id=file_row.id,
                    material_id=material.material_id,
                    mat_type=material.mat_type,
                    parse_status=material.parse_status,
                    line_start=material.raw.line_start,
                    line_end=material.raw.line_end,
                    raw_hash=material.raw.raw_hash,
                )
            )
        for contact in parsed.contacts:
            session.add(
                CaeContact(
                    deck_id=deck.id,
                    file_id=file_row.id,
                    contact_type=contact.contact_type,
                    ssid=contact.ssid,
                    msid=contact.msid,
                    parse_status=contact.parse_status,
                    line_start=contact.raw.line_start,
                    line_end=contact.raw.line_end,
                    raw_hash=contact.raw.raw_hash,
                )
            )
        for control in parsed.controls:
            session.add(
                CaeControl(
                    deck_id=deck.id,
                    file_id=file_row.id,
                    control_type=control.control_type,
                    parse_status=control.parse_status,
                    line_start=control.raw.line_start,
                    line_end=control.raw.line_end,
                    raw_hash=control.raw.raw_hash,
                )
            )
        for database in parsed.databases:
            session.add(
                CaeDatabase(
                    deck_id=deck.id,
                    file_id=file_row.id,
                    database_type=database.database_type,
                    dt=database.dt,
                    parse_status=database.parse_status,
                    line_start=database.raw.line_start,
                    line_end=database.raw.line_end,
                    raw_hash=database.raw.raw_hash,
                )
            )

    for edge in graph.edges:
        if edge.parent not in file_rows:
            continue  # edge from a deck we weren't given the parsed content for
        resolved_row = file_rows.get(edge.resolved_target) if edge.resolved_target else None
        session.add(
            CaeInclude(
                deck_id=deck.id,
                parent_file_id=file_rows[edge.parent].id,
                target_as_written=edge.target_as_written,
                resolved_file_id=resolved_row.id if resolved_row else None,
                status=edge.status,
            )
        )

    session.commit()
    return deck
