"""Structured CAE retrieval — TRD_LEVEL3.md §18/§23, PRD_LEVEL3.md §14.

Answers questions like "which material is used by Part 1042?" from the
parsed `cae_*` entity tables directly — never guessed from text similarity
(TRD_LEVEL3.md §18: "This should be answered from structured data, not
guessed from semantic similarity").

Every query is scoped by `deck_id`: part/material/section ids are only
unique within one deck's own numbering, not globally across the corpus.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from packages.domain.cae import (
    CaeContact,
    CaeControl,
    CaeDatabase,
    CaeDeck,
    CaeInclude,
    CaeMaterial,
    CaePart,
    CaeSection,
)


def find_deck_by_key(session: Session, deck_key: str) -> CaeDeck | None:
    return session.query(CaeDeck).filter_by(deck_key=deck_key).one_or_none()


def get_materials_for_part(session: Session, deck_id: uuid.UUID, part_id: int) -> list[CaeMaterial]:
    """ "Which material is used by Part 1042?" — TRD_LEVEL3.md §18's own example."""
    part = session.query(CaePart).filter_by(deck_id=deck_id, part_id=part_id).first()
    if part is None or part.material_id is None:
        return []
    return session.query(CaeMaterial).filter_by(deck_id=deck_id, material_id=part.material_id).all()


def get_section_for_part(session: Session, deck_id: uuid.UUID, part_id: int) -> CaeSection | None:
    part = session.query(CaePart).filter_by(deck_id=deck_id, part_id=part_id).first()
    if part is None or part.section_id is None:
        return None
    return session.query(CaeSection).filter_by(deck_id=deck_id, section_id=part.section_id).first()


def get_parts_using_material(session: Session, deck_id: uuid.UUID, material_id: int) -> list[CaePart]:
    return session.query(CaePart).filter_by(deck_id=deck_id, material_id=material_id).all()


def get_contacts_in_deck(session: Session, deck_id: uuid.UUID) -> list[CaeContact]:
    return session.query(CaeContact).filter_by(deck_id=deck_id).all()


def get_control_cards_in_deck(session: Session, deck_id: uuid.UUID) -> list[CaeControl]:
    return session.query(CaeControl).filter_by(deck_id=deck_id).all()


def get_database_outputs_in_deck(session: Session, deck_id: uuid.UUID) -> list[CaeDatabase]:
    return session.query(CaeDatabase).filter_by(deck_id=deck_id).all()


def get_includes_for_deck(session: Session, deck_id: uuid.UUID) -> list[CaeInclude]:
    return session.query(CaeInclude).filter_by(deck_id=deck_id).all()


def get_all_parts(session: Session, deck_id: uuid.UUID) -> list[CaePart]:
    return session.query(CaePart).filter_by(deck_id=deck_id).all()
