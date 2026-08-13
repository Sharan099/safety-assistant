"""Structured + Hybrid RAG — TRD.md §19/§22, TRD_LEVEL3.md §15/§20-25.

"Parent-child expansion" (TRD.md §19) is implemented via the chunk's
containing `DocumentSection`, not a separate parent-chunk hierarchy:
`packages/ingestion` produces single-level (leaf-only) chunks, so the
section a chunk belongs to is the natural "parent" broader-context unit —
see `RetrievedChunk.section_content`.

Structured CAE search (`packages/retrieval/structured.py`) is deliberately
NOT fused into this module's RRF ranking. PRD_LEVEL3.md §14 states outright:
"Structured search is complementary to RAG" — a `CaePart`/`CaeMaterial` row
has no principled way to share a rank with a text chunk's BM25/dense score,
and the existing tool list (PRD_LEVEL3.md §26) already keeps
`retrieve_knowledge` and `retrieve_structured_cae` as two separate tools.
The retrieval pipeline this module implements is:

    BM25 (packages/retrieval/bm25.py, real BM25Okapi — docs/ADR/0012)
        +
    Dense (pgvector, packages/retrieval/embeddings.py)
        v
    RRF (reciprocal_rank_fusion)
        v
    Reranker (packages/retrieval/rerank.py)
        v
    Authority/relevance/dedup guard (packages/retrieval/relevance.py) —
    CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 8: "Do not expose raw top-k
    chunks."
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

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
from packages.retrieval.bm25 import Bm25Index, bm25_search, build_bm25_index
from packages.retrieval.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from packages.retrieval.relevance import DEFAULT_MIN_SHARED_TERMS, has_known_authority, is_relevant
from packages.retrieval.rerank import LexicalAuthorityReranker, RerankCandidate, Reranker, rerank

RRF_K = 60
DEFAULT_LIMIT = 10
CANDIDATE_MULTIPLIER = 4  # fetch this many x `limit` from each retrieval leg before fusing
MAX_CHUNKS_PER_DOCUMENT = 3  # dedup: cap how much of the top-k one document can dominate


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
    rerank_score: float
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


def full_text_search(
    session: Session, query_text: str, filters: SourceFilter, *, limit: int, bm25_index: Bm25Index | None = None
) -> list[Any]:
    """The BM25 leg (docs/ADR/0012 — replaces the earlier `ts_rank`
    approximation, docs/ADR/0009). `bm25_index` lets a caller reuse one
    index across many queries (packages/retrieval eval harness) instead of
    rebuilding it per call; `retrieve()` builds one itself when not given."""
    index = bm25_index if bm25_index is not None else build_bm25_index(session)
    ranked_chunk_ids = bm25_search(index, query_text, limit=limit)
    if not ranked_chunk_ids:
        return []

    rows = _base_query(session, filters).filter(DocumentChunk.id.in_(ranked_chunk_ids)).all()
    # SQL `IN` doesn't preserve order — restore the real BM25 ranking, and
    # drop any id a filter (source_type/authority_level/document_key)
    # excluded from `rows`.
    row_by_chunk_id = {row[0].id: row for row in rows}
    return [row_by_chunk_id[cid] for cid in ranked_chunk_ids if cid in row_by_chunk_id]


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


def _passes_guard(chunk: RetrievedChunk, query_text: str, *, min_shared_terms: int) -> bool:
    if not has_known_authority(chunk.authority_level):
        return False
    return is_relevant(query_text, chunk.content, min_shared_terms=min_shared_terms)


def _deduplicate(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen_content: set[str] = set()
    per_document: dict[str, int] = {}
    deduped: list[RetrievedChunk] = []
    for r in results:
        if r.content in seen_content:
            continue
        if per_document.get(r.document_key, 0) >= MAX_CHUNKS_PER_DOCUMENT:
            continue
        seen_content.add(r.content)
        per_document[r.document_key] = per_document.get(r.document_key, 0) + 1
        deduped.append(r)
    return deduped


def retrieve(
    session: Session,
    query_text: str,
    *,
    filters: SourceFilter | None = None,
    limit: int = DEFAULT_LIMIT,
    provider: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    bm25_index: Bm25Index | None = None,
    min_shared_terms: int = DEFAULT_MIN_SHARED_TERMS,
) -> list[RetrievedChunk]:
    """BM25 + dense -> RRF -> reranker -> relevance/authority/dedup guard
    (packages/retrieval/relevance.py) -> never raw top-k chunks
    (CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 8)."""
    filters = filters or SourceFilter()
    reranker = reranker or LexicalAuthorityReranker()
    candidate_limit = max(limit * CANDIDATE_MULTIPLIER, 20)

    fts_rows = full_text_search(session, query_text, filters, limit=candidate_limit, bm25_index=bm25_index)
    vector_rows = vector_search(session, query_text, filters, limit=candidate_limit, provider=provider)

    fts_ids = [row[0].id for row in fts_rows]
    vector_ids = [row[0].id for row in vector_rows]
    fused_scores = reciprocal_rank_fusion([fts_ids, vector_ids])

    row_by_id: dict[uuid.UUID, Any] = {}
    for row in (*fts_rows, *vector_rows):
        row_by_id.setdefault(row[0].id, row)

    fts_id_set, vector_id_set = set(fts_ids), set(vector_ids)
    # Rank every fused candidate, not just the first `limit` — filtering
    # happens after ranking, so a naive pre-filter cutoff would starve the
    # result set even when enough relevant candidates exist further down.
    fused_ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    candidates: list[RetrievedChunk] = []
    for chunk_id in fused_ranked_ids:
        chunk, revision, document, knowledge_source, section = row_by_id[chunk_id]
        locator = chunk.source_locator or {}
        candidates.append(
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
                rerank_score=fused_scores[chunk_id],  # overwritten below once reranked
                matched_fts=chunk_id in fts_id_set,
                matched_vector=chunk_id in vector_id_set,
            )
        )

    if candidates:
        rerank_candidates = [
            RerankCandidate(
                id=c.chunk_id, content=c.content, authority_level=c.authority_level, fused_score=c.fused_score
            )
            for c in candidates
        ]
        observation = rerank(reranker, query_text, rerank_candidates)
        rerank_score_by_id = {r.id: r.rerank_score for r in observation.results}
        for c in candidates:
            c.rerank_score = rerank_score_by_id.get(c.chunk_id, c.fused_score)
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)

    relevant = [c for c in candidates if _passes_guard(c, query_text, min_shared_terms=min_shared_terms)]
    return _deduplicate(relevant)[:limit]
