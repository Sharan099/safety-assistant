"""Reranker — TRD_LEVEL3.md §20/§25, applied after RRF fusion, before the
relevance/authority gate.

`LexicalAuthorityReranker` remains the **production default** —
`CrossEncoderReranker` (docs/ADR/0015) is implemented, real, and
benchmarked (`evals/reranker_benchmark.py`), but rejected as the default on
this machine: it genuinely improves ranking quality (NDCG@10 0.954 vs
0.938) but measured at ~3.5s/query on this CPU (i5-8250U) — PASSIVE_SAFETY_
LEVEL3_FINAL_FIX.md §21's evaluation gates require a candidate to clear
*quality, latency, memory, and storage together*, not quality alone, and
3.5s per rerank step is not "remain practical on the development machine"
for an interactive investigation Copilot. `CrossEncoderReranker` stays
available (real, tested, swappable via the same `Reranker` Protocol) for
contexts where that latency is acceptable — offline evaluation, or a future
deployment with faster hardware — rather than deleted.

`LexicalAuthorityReranker`'s score is the fused RRF score plus a small
bonus for literal query-term overlap (rewards exact engineering-identifier
matches like `*MAT_024`/`HIC15`) plus a small authority-tier bonus.
`CrossEncoderReranker`'s score is the model's own raw relevance logit for
(query, candidate) — a full re-score, not an additive adjustment, matching
standard cross-encoder reranking architecture (the doc's own diagram:
"RRF -> top 30-50 -> cross-encoder -> top 5-10"). Neither ever reorders
relevance/authority filtering — that gate still runs after this, unchanged.
"""

from __future__ import annotations

import functools
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


# Chosen by evals/reranker_benchmark.py (docs/ADR/0015): small, real,
# CPU-friendly cross-encoder — same substitution rationale as
# docs/ADR/0014 for the doc's own suggested (but much larger) candidates.
DEFAULT_CROSS_ENCODER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """The real cross-encoder tier — see module docstring."""

    model_version = "v1"

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> None:
        # Local import: avoids paying ONNX Runtime's import cost for callers
        # that only ever construct LexicalAuthorityReranker.
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query_text: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        if not candidates:
            return []
        scores = self._model.rerank(query_text, [c.content for c in candidates])
        return [RerankResult(id=c.id, rerank_score=float(s)) for c, s in zip(candidates, scores, strict=True)]


@functools.lru_cache(maxsize=1)
def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """A process-wide singleton for the opt-in high-quality/slow tier — not
    called by retrieve()'s default path (see module docstring: rejected on
    latency, not offered as the default). Available for a caller that has
    explicitly decided the ~3.5s/query cost is acceptable (offline
    evaluation, a future faster-hardware deployment)."""
    return CrossEncoderReranker()


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
