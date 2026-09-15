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
from safety_assistant.retrieval.authz import anonymous_sql
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
    stmt = stmt.where(scope.authz.sql(focus=False) if scope.authz is not None else anonymous_sql())
    # "Ask this document" plus the regulations the question names: an engineer comparing a test
    # report with UN R94 needs both; neither restriction widens authorization above.
    focus = scope.authz.document_ids if scope.authz is not None else ()
    if focus and scope.regulation_keys:
        stmt = stmt.where(or_(Regulation.id.in_(focus), regulation_key_matches(scope.regulation_keys)))
    elif focus:
        stmt = stmt.where(Regulation.id.in_(focus))
    elif scope.regulation_keys:
        stmt = stmt.where(regulation_key_matches(scope.regulation_keys))
    if scope.version_ids:
        stmt = stmt.where(RegulationVersion.id.in_([uuid.UUID(v) for v in scope.version_ids]))
    else:
        # Temporal validity: a NULL valid_from means "validity unknown" (supporting
        # documents), which is treated as always in force rather than never.
        d = scope.effective_date(today)
        stmt = stmt.where(or_(RegulationVersion.valid_from.is_(None), RegulationVersion.valid_from <= d))
        stmt = stmt.where(or_(RegulationVersion.valid_to.is_(None), RegulationVersion.valid_to > d))
    if scope.kinds:
        stmt = stmt.where(Regulation.kind.in_(list(scope.kinds)))
    if scope.authority_levels:
        stmt = stmt.where(Regulation.authority_level.in_(list(scope.authority_levels)))
    return stmt


def regulation_key_matches(keys: tuple[str, ...]):  # type: ignore[no-untyped-def]
    """A regulation key selects the text *and* its supplements/amendment sheets (`UN-R94-AMEND-05`)."""
    return or_(
        *[Regulation.regulation_key == k for k in keys], *[Regulation.regulation_key.like(f"{k}-%") for k in keys]
    )


def key_in_scope(regulation_key: str, keys: tuple[str, ...]) -> bool:
    return any(regulation_key == k or regulation_key.startswith(f"{k}-") for k in keys)


def document_in_scope(scope: ScopeFilter, document_id: uuid.UUID, regulation_key: str) -> bool:
    """In-memory twin of the focus/keys clause of `scoped_statement` (BM25 leg)."""
    focus = scope.authz.document_ids if scope.authz is not None else ()
    if focus and document_id in focus:
        return True
    if scope.regulation_keys:
        return key_in_scope(regulation_key, scope.regulation_keys)
    return not focus


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
