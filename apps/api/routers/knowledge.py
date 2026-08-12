"""Knowledge Retrieval — APP_FLOW.md §14, PRD.md PR-011."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from packages.retrieval.search import RetrievedChunk, SourceFilter, retrieve

router = APIRouter(tags=["knowledge"])


@router.get("/knowledge/search", response_model=list[RetrievedChunk])
def search_knowledge(
    q: str = Query(..., min_length=1),
    source_type: str | None = None,
    authority_level: str | None = None,
    document_key: str | None = None,
    limit: int = 10,
    session: Session = Depends(get_db),
) -> list[RetrievedChunk]:
    filters = SourceFilter(source_type=source_type, authority_level=authority_level, document_key=document_key)
    return retrieve(session, q, filters=filters, limit=limit)
