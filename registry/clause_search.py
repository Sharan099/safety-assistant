"""Search-and-select resolvable regulation clauses from the live corpus."""

from __future__ import annotations

from sqlalchemy import or_
from sqlalchemy.orm import Session

from database.models import Chunk, Document, Regulation
from registry.harness_limits import extract_limit_details


def search_resolvable_clauses(
    db: Session,
    *,
    q: str,
    regulation_code: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return clause rows that exist in DB and have parseable injury limits."""
    term = (q or "").strip()
    if len(term) < 2:
        return []

    query = (
        db.query(Chunk, Document, Regulation)
        .join(Document, Chunk.document_id == Document.id)
        .join(Regulation, Document.regulation_id == Regulation.id)
        .filter(Chunk.section.isnot(None), Chunk.section != "")
    )
    if regulation_code:
        query = query.filter(Regulation.regulation_code == regulation_code)

    like = f"%{term}%"
    query = query.filter(
        or_(
            Chunk.chunk_text.ilike(like),
            Chunk.section.ilike(like),
            Regulation.regulation_code.ilike(like),
        )
    )
    rows = query.order_by(Regulation.regulation_code, Chunk.section).limit(limit * 3).all()

    seen: set[str] = set()
    results: list[dict] = []
    for chunk, doc, reg in rows:
        if not chunk.section:
            continue
        linked = f"{reg.regulation_code}#{chunk.section}"
        if linked in seen:
            continue
        # Must resolve at least one common criterion for harness linking
        resolvable = False
        for crit in ("ThCC", "HIC36", "HIC", "TCFC"):
            if extract_limit_details(crit, chunk.chunk_text) is not None:
                resolvable = True
                break
        if not resolvable:
            continue
        seen.add(linked)
        snippet = (chunk.chunk_text or "")[:280].replace("\n", " ")
        results.append(
            {
                "regulation_code": reg.regulation_code,
                "section": chunk.section,
                "document_name": doc.document_name,
                "linked_regulation_clause": linked,
                "snippet": snippet,
            }
        )
        if len(results) >= limit:
            break
    return results


def validate_linked_clause(db: Session, linked_regulation_clause: str) -> bool:
    if not linked_regulation_clause or "#" not in linked_regulation_clause:
        return False
    reg_code, section = linked_regulation_clause.split("#", 1)
    chunk = (
        db.query(Chunk)
        .join(Document)
        .join(Regulation)
        .filter(Regulation.regulation_code == reg_code, Chunk.section == section)
        .first()
    )
    if not chunk:
        return False
    chunk_text = str(chunk.chunk_text or "")
    return any(extract_limit_details(crit, chunk_text) is not None for crit in ("ThCC", "HIC36", "HIC", "TCFC"))
