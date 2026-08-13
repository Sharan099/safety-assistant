"""Reranker — TRD_LEVEL3.md §20/§25, applied after RRF fusion, before the
relevance/authority gate.

A real cross-encoder (`sentence-transformers`) is deferred (`docs/ADR/0011`:
12 GB free disk, `torch`/`transformers` resolve to several GB — a real
risk, not a style preference). `LexicalAuthorityReranker` is a dependency-
free stand-in exercising the real architecture slot (RRF -> Reranker ->
Gate) so nothing downstream has to change when a real model is installed —
same `Protocol`-swap pattern as `docs/ADR/0007`'s `HashingEmbeddingProvider`.

Score = the fused RRF score, plus a small bonus for literal query-term
overlap with the chunk's content (rewards exact engineering-identifier
matches like `*MAT_024`/`HIC15` that a hashed/semantic signal alone won't
reliably privilege), plus a small authority-tier bonus (AUTHORITATIVE /
OFFICIAL_DOCUMENTATION content edges out REFERENCE/SYNTHETIC content when
otherwise close). Never reorders relevance/authority filtering — that gate
still runs after this, unchanged.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from packages.retrieval.relevance import significant_tokens

# BACKEND_SCHEMA.md §18 KnowledgeSource.authority_level values.
_AUTHORITY_BONUS = {
    "AUTHORITATIVE": 0.02,
    "OFFICIAL_DOCUMENTATION": 0.015,
    "INTERNAL_APPROVED": 0.01,
    "HISTORICAL": 0.005,
    "REFERENCE": 0.0,
    "SYNTHETIC": 0.0,
}
_MAX_OVERLAP_TERMS_COUNTED = 5
_OVERLAP_BONUS_PER_TERM = 0.01


@dataclass
class RerankCandidate:
    id: uuid.UUID
    content: str
    authority_level: str
    fused_score: float


@dataclass
class RerankResult:
    id: uuid.UUID
    rerank_score: float


class Reranker(Protocol):
    model_name: str
    model_version: str

    def score(self, query_text: str, candidates: list[RerankCandidate]) -> list[RerankResult]: ...


class LexicalAuthorityReranker:
    model_name = "lexical-authority-heuristic"
    model_version = "v1"

    def score(self, query_text: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        query_terms = significant_tokens(query_text)
        results = []
        for c in candidates:
            overlap = len(query_terms & significant_tokens(c.content))
            overlap_bonus = min(overlap, _MAX_OVERLAP_TERMS_COUNTED) * _OVERLAP_BONUS_PER_TERM
            authority_bonus = _AUTHORITY_BONUS.get(c.authority_level, 0.0)
            results.append(RerankResult(id=c.id, rerank_score=c.fused_score + overlap_bonus + authority_bonus))
        return results


@dataclass
class RerankObservation:
    """TRD_LEVEL3.md §28 retrieval observability: model/version + latency,
    alongside each candidate's score, for evaluation (packages/retrieval
    eval harness) and debugging."""

    model_name: str
    model_version: str
    latency_ms: float
    results: list[RerankResult]


def rerank(reranker: Reranker, query_text: str, candidates: list[RerankCandidate]) -> RerankObservation:
    start = time.perf_counter()
    results = reranker.score(query_text, candidates)
    latency_ms = (time.perf_counter() - start) * 1000
    results.sort(key=lambda r: r.rerank_score, reverse=True)
    return RerankObservation(
        model_name=reranker.model_name, model_version=reranker.model_version, latency_ms=latency_ms, results=results
    )
