"""Shared retrieval plumbing: the scoped candidate universe and the row model.

Scope (lifecycle status, temporal validity, regulation, data class) is
applied in SQL *before* any leg scores anything — historical queries filter
before ranking (CLAUDE.md §8)."""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from safety_assistant.domain.regulations import RETRIEVABLE_CURRENT, RETRIEVABLE_HISTORICAL
from safety_assistant.persistence.models import Chunk, Regulation, RegulationVersion, Section
from safety_assistant.retrieval.filters import ScopeFilter


@dataclass
class CandidateRow:
    chunk: Chunk
    section: Section
    version: RegulationVersion
    regulation: Regulation


def scoped_statement(
    scope: ScopeFilter, *, today: datetime.date | None = None
) -> Select[tuple[Chunk, Section, RegulationVersion, Regulation]]:
    stmt = (
        select(Chunk, Section, RegulationVersion, Regulation)
        .join(Section, Section.id == Chunk.section_id)
        .join(RegulationVersion, RegulationVersion.id == Chunk.version_id)
        .join(Regulation, Regulation.id == RegulationVersion.regulation_id)
    )
    statuses = RETRIEVABLE_HISTORICAL if (scope.include_superseded or scope.as_of) else RETRIEVABLE_CURRENT
    stmt = stmt.where(RegulationVersion.status.in_([s.value for s in statuses]))
    stmt = stmt.where(Regulation.data_class.in_(list(scope.data_classes)))
    if scope.version_ids:
        stmt = stmt.where(RegulationVersion.id.in_([uuid.UUID(v) for v in scope.version_ids]))
    else:
        # Temporal validity: a NULL valid_from means "validity unknown" (supporting
        # documents), which is treated as always in force rather than never.
        d = scope.effective_date(today)
        stmt = stmt.where(or_(RegulationVersion.valid_from.is_(None), RegulationVersion.valid_from <= d))
        stmt = stmt.where(or_(RegulationVersion.valid_to.is_(None), RegulationVersion.valid_to > d))
    if scope.regulation_keys:
        stmt = stmt.where(Regulation.regulation_key.in_(list(scope.regulation_keys)))
    if scope.kinds:
        stmt = stmt.where(Regulation.kind.in_(list(scope.kinds)))
    if scope.authority_levels:
        stmt = stmt.where(Regulation.authority_level.in_(list(scope.authority_levels)))
    return stmt


def scoped_chunk_ids(session: Session, scope: ScopeFilter, *, today: datetime.date | None = None) -> list[uuid.UUID]:
    stmt = scoped_statement(scope, today=today).with_only_columns(Chunk.id)
    return list(session.scalars(stmt).all())


def rows_by_id(
    session: Session, scope: ScopeFilter, ids: list[uuid.UUID], *, today: datetime.date | None = None
) -> dict[uuid.UUID, CandidateRow]:
    if not ids:
        return {}
    stmt = scoped_statement(scope, today=today).where(Chunk.id.in_(ids))
    return {c.id: CandidateRow(c, s, v, r) for c, s, v, r in session.execute(stmt).all()}
