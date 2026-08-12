"""Baseline RAG: PostgreSQL FTS + pgvector + metadata filters + RRF —
TRD.md §19/§22, CLAUDE_CODE_BOOTSTRAP_PROMPT.md §14.

"Parent-child expansion" (TRD.md §19) is implemented via the chunk's
containing `DocumentSection`, not a separate parent-chunk hierarchy:
`packages/ingestion` produces single-level (leaf-only) chunks, so the
section a chunk belongs to is the natural "parent" broader-context unit —
see `RetrievedChunk.section_content`.

No reranker (TRD.md §21: benchmark BM25+dense+RRF first, add one only if
evaluation shows a real need — packages/retrieval/eval.py is that benchmark).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.orm import Session

from packages.domain.knowledge import (
    Document,
    DocumentChunk,
    DocumentRevision,
    DocumentSection,
    Embedding,
    KnowledgeSource,
)
from packages.retrieval.embeddings import EmbeddingProvider, HashingEmbeddingProvider

RRF_K = 60
DEFAULT_LIMIT = 10
CANDIDATE_MULTIPLIER = 4  # fetch this many x `limit` from each retrieval leg before fusing


@dataclass
class SourceFilter:
    source_type: str | None = None
    authority_level: str | None = None
    document_key: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: uuid.UUID
    content: str
    document_key: str
    document_title: str
    authority_level: str
    source_type: str
    revision_label: str
    section_title: str | None
    section_content: str | None
    page_start: int | None
    page_end: int | None
    fused_score: float
    matched_fts: bool
    matched_vector: bool


def _base_query(session: Session, filters: SourceFilter):  # type: ignore[no-untyped-def]
    q = (
        session.query(DocumentChunk, DocumentRevision, Document, KnowledgeSource, DocumentSection)
        .join(DocumentRevision, DocumentChunk.document_revision_id == DocumentRevision.id)
        .join(Document, DocumentRevision.document_id == Document.id)
        .join(KnowledgeSource, Document.knowledge_source_id == KnowledgeSource.id)
        .outerjoin(DocumentSection, DocumentChunk.section_id == DocumentSection.id)
    )
    if filters.source_type:
        q = q.filter(KnowledgeSource.source_type == filters.source_type)
    if filters.authority_level:
        q = q.filter(KnowledgeSource.authority_level == filters.authority_level)
    if filters.document_key:
        q = q.filter(Document.document_key == filters.document_key)
    return q


def full_text_search(session: Session, query_text: str, filters: SourceFilter, *, limit: int) -> list[Any]:
    tsquery = sa.func.plainto_tsquery("english", query_text)
    tsvector = sa.func.to_tsvector("english", DocumentChunk.content)
    rank = sa.func.ts_rank(tsvector, tsquery)
    q = _base_query(session, filters).filter(tsvector.op("@@")(tsquery)).order_by(rank.desc()).limit(limit)
    return q.all()  # type: ignore[no-any-return]  # _base_query is untyped (SQLAlchemy query-builder chain)


def vector_search(
    session: Session, query_text: str, filters: SourceFilter, *, limit: int, provider: EmbeddingProvider | None = None
) -> list[Any]:
    provider = provider or HashingEmbeddingProvider()
    query_vector = provider.embed(query_text)
    distance = Embedding.embedding.cosine_distance(query_vector)
    q = (
        _base_query(session, filters)
        .join(Embedding, Embedding.chunk_id == DocumentChunk.id)
        .filter(Embedding.model_name == provider.model_name, Embedding.model_version == provider.model_version)
        .order_by(distance.asc())
        .limit(limit)
    )
    return q.all()  # type: ignore[no-any-return]


def reciprocal_rank_fusion(ranked_id_lists: list[list[uuid.UUID]], *, k: int = RRF_K) -> dict[uuid.UUID, float]:
    scores: dict[uuid.UUID, float] = {}
    for ranked in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def retrieve(
    session: Session,
    query_text: str,
    *,
    filters: SourceFilter | None = None,
    limit: int = DEFAULT_LIMIT,
    provider: EmbeddingProvider | None = None,
) -> list[RetrievedChunk]:
    filters = filters or SourceFilter()
    candidate_limit = max(limit * CANDIDATE_MULTIPLIER, 20)

    fts_rows = full_text_search(session, query_text, filters, limit=candidate_limit)
    vector_rows = vector_search(session, query_text, filters, limit=candidate_limit, provider=provider)

    fts_ids = [row[0].id for row in fts_rows]
    vector_ids = [row[0].id for row in vector_rows]
    fused_scores = reciprocal_rank_fusion([fts_ids, vector_ids])

    row_by_id: dict[uuid.UUID, Any] = {}
    for row in (*fts_rows, *vector_rows):
        row_by_id.setdefault(row[0].id, row)

    fts_id_set, vector_id_set = set(fts_ids), set(vector_ids)
    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:limit]

    results = []
    for chunk_id in ranked_ids:
        chunk, revision, document, knowledge_source, section = row_by_id[chunk_id]
        locator = chunk.source_locator or {}
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                content=chunk.content,
                document_key=document.document_key,
                document_title=document.title,
                authority_level=knowledge_source.authority_level,
                source_type=knowledge_source.source_type,
                revision_label=revision.revision_label,
                section_title=section.title if section else None,
                section_content=section.content if section else None,
                page_start=locator.get("page_start"),
                page_end=locator.get("page_end"),
                fused_score=fused_scores[chunk_id],
                matched_fts=chunk_id in fts_id_set,
                matched_vector=chunk_id in vector_id_set,
            )
        )
    return results
