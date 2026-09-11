"""Evidence packaging: stable evidence IDs, parent-section expansion, bounded
cross-reference expansion, context budgeting.

Every `Evidence` carries the full provenance chain (chunk → section → version
→ artifact) so a citation can be validated and opened without another lookup.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from safety_assistant.ingestion.chunk import estimate_tokens
from safety_assistant.persistence.models import CrossReference, Section, SourceArtifact
from safety_assistant.retrieval.base import CandidateRow

MAX_PARENT_CHARS = 1200
MAX_RELATED_PER_EVIDENCE = 2
MAX_RELATED_CHARS = 600


class RelatedSection(BaseModel):
    path: str
    citation_label: str
    excerpt: str
    via: str  # the raw cross-reference text


class LegRanks(BaseModel):
    dense: int | None = None
    sparse: int | None = None
    exact: int | None = None
    fused_score: float
    rerank_score: float | None = None


class Evidence(BaseModel):
    evidence_id: str
    chunk_id: uuid.UUID
    regulation_key: str
    regulation_title: str
    kind: str
    jurisdiction: str
    authority_level: str
    data_class: str = "PUBLIC"
    version_id: uuid.UUID
    version_label: str
    version_status: str
    valid_from: datetime.date | None
    valid_to: datetime.date | None
    published_at: datetime.date | None
    section_id: uuid.UUID
    section_path: str
    section_number: str | None
    section_title: str | None
    annex: str | None
    normative: bool | None
    chunk_type: str
    page_start: int | None
    page_end: int | None
    citation_label: str
    content: str
    parent_context: str | None = None
    related: list[RelatedSection] = []
    ranks: LegRanks
    source_sha256: str
    source_uri: str | None
    storage_uri: str
    token_count: int


class EvidenceBundle(BaseModel):
    evidence: list[Evidence]
    total_tokens: int
    truncated: bool  # budget cut some candidates


def build_evidence(
    session: Session,
    ranked: list[tuple[CandidateRow, LegRanks]],
    *,
    token_budget: int,
    expand_parents: bool = True,
    expand_cross_refs: bool = True,
) -> EvidenceBundle:
    artifacts = _artifacts(session, [r.version.source_artifact_id for r, _ in ranked])
    evidence: list[Evidence] = []
    total = 0
    truncated = False
    for i, (row, ranks) in enumerate(ranked, start=1):
        parent = _parent_context(session, row.section) if expand_parents else None
        related = _related(session, row, prefix=_prefix(row)) if expand_cross_refs else []
        tokens = (
            estimate_tokens(row.chunk.content)
            + (estimate_tokens(parent) if parent else 0)
            + sum(estimate_tokens(r.excerpt) for r in related)
        )
        if evidence and total + tokens > token_budget:
            truncated = True
            break
        total += tokens
        art = artifacts[row.version.source_artifact_id]
        evidence.append(
            Evidence(
                evidence_id=f"E{i}",
                chunk_id=row.chunk.id,
                regulation_key=row.regulation.regulation_key,
                regulation_title=row.regulation.title,
                kind=row.regulation.kind,
                jurisdiction=row.regulation.jurisdiction,
                authority_level=row.regulation.authority_level,
                data_class=row.regulation.data_class,
                version_id=row.version.id,
                version_label=row.version.version_label,
                version_status=row.version.status,
                valid_from=row.version.valid_from,
                valid_to=row.version.valid_to,
                published_at=row.version.published_at,
                section_id=row.section.id,
                section_path=row.section.path,
                section_number=row.section.section_number,
                section_title=row.section.title,
                annex=row.section.annex,
                normative=row.section.normative,
                chunk_type=row.chunk.chunk_type,
                page_start=row.chunk.page_start,
                page_end=row.chunk.page_end,
                citation_label=row.chunk.citation_label,
                content=row.chunk.content,
                parent_context=parent,
                related=related,
                ranks=ranks,
                source_sha256=art.sha256,
                source_uri=art.source_uri,
                storage_uri=art.storage_uri,
                token_count=tokens,
            )
        )
    return EvidenceBundle(evidence=evidence, total_tokens=total, truncated=truncated)


def _prefix(row: CandidateRow) -> str:
    return f"{row.regulation.regulation_key.replace('-', ' ')} {row.version.version_label.split(' ')[0]}"


def _artifacts(session: Session, ids: list[uuid.UUID]) -> dict[uuid.UUID, SourceArtifact]:
    if not ids:
        return {}
    rows = session.scalars(select(SourceArtifact).where(SourceArtifact.id.in_(set(ids)))).all()
    return {a.id: a for a in rows}


def _parent_context(session: Session, section: Section) -> str | None:
    """Title chain + the parent section's own text (trimmed) — the broader
    unit a fine-grained clause lives in."""
    if section.parent_section_id is None:
        return None
    parent = session.get(Section, section.parent_section_id)
    if parent is None:
        return None
    head = " ".join(x for x in (parent.section_number, parent.title) if x) or parent.path
    body = parent.content[:MAX_PARENT_CHARS]
    return f"{head}\n{body}".strip() if body else head


def _related(session: Session, row: CandidateRow, *, prefix: str) -> list[RelatedSection]:
    refs = session.scalars(
        select(CrossReference)
        .where(CrossReference.from_section_id == row.section.id, CrossReference.resolved_section_id.is_not(None))
        .limit(MAX_RELATED_PER_EVIDENCE)
    ).all()
    out: list[RelatedSection] = []
    for ref in refs:
        target = session.get(Section, ref.resolved_section_id) if ref.resolved_section_id else None
        if target is None or not target.content:
            continue
        where = f"§{target.section_number.rstrip('.')}" if target.section_number else target.path
        if target.annex:
            where = f"{target.annex} {where}"
        pages = f" (p. {target.page_start})" if target.page_start else ""
        out.append(
            RelatedSection(
                path=target.path,
                citation_label=f"{prefix} {where}{pages}",
                excerpt=target.content[:MAX_RELATED_CHARS],
                via=ref.raw_text,
            )
        )
    return out


def evidence_to_dict(e: Evidence) -> dict[str, Any]:
    return e.model_dump(mode="json")
