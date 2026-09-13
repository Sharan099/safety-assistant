"""Hybrid retrieval service — the deterministic production baseline (CLAUDE.md §8).

query → scope parse → SQL scope (status/temporal/regulation/data class)
      → dense top-k + sparse top-k (+ exact-identifier leg)
      → RRF → dedup/diversify → rerank → relevance/authority guard
      → parent + cross-ref expansion → evidence bundle within a token budget
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import or_
from sqlalchemy.orm import Session

from safety_assistant.config import Settings, get_settings
from safety_assistant.domain.temporal import QueryScope, parse_query_scope
from safety_assistant.observability import metrics, span
from safety_assistant.persistence.models import Chunk, Section
from safety_assistant.providers.embeddings import EmbeddingProvider, get_embedding_provider
from safety_assistant.providers.rerankers import RerankCandidate, Reranker, get_reranker
from safety_assistant.retrieval.base import CandidateRow, rows_by_id, scoped_statement
from safety_assistant.retrieval.context import EvidenceBundle, LegRanks, build_evidence
from safety_assistant.retrieval.dense import dense_search
from safety_assistant.retrieval.filters import DEFAULT_MIN_SHARED_TERMS, ScopeFilter, has_known_authority, is_relevant
from safety_assistant.retrieval.fusion import reciprocal_rank_fusion
from safety_assistant.retrieval.rerank import apply_reranker
from safety_assistant.retrieval.sparse import Bm25Index, sparse_search

log = logging.getLogger(__name__)

MAX_CHUNKS_PER_VERSION = 4  # diversification: one document may not fill the whole list
EXACT_LEG_WEIGHT = 2.0  # an exact clause-id hit is stronger evidence than any single semantic leg
DEFAULT_TOKEN_BUDGET = 6000


@dataclass
class RetrievalConfig:
    dense_top_k: int = 30
    sparse_top_k: int = 30
    final_k: int = 10
    use_dense: bool = True
    use_sparse: bool = True
    use_exact: bool = True
    use_reranker: bool = True
    # RRF leg weights (ranks only; a weight scales one leg's 1/(k+rank) contribution).
    dense_weight: float = 0.75  # tuned 2026-09-12 on regulatory_v1+v2 (docs/evaluation.md)
    sparse_weight: float = 1.0
    # Rerank only the top-N fused candidates (None = all). Bounds cross-encoder latency; the heuristic
    # reranker is cheap enough to score everything.
    rerank_top_n: int | None = 12
    # "always" runs the reranker on every query. "adaptive" skips it when the legs already agree:
    # the query names a clause identifier and the exact leg found it, or dense and sparse put the same
    # chunk first. No tuned thresholds — measured in docs/evaluation.md before this became an option.
    rerank_policy: Literal["always", "adaptive"] = "always"
    expand_parents: bool = True
    expand_cross_refs: bool = True
    min_shared_terms: int = DEFAULT_MIN_SHARED_TERMS
    token_budget: int = DEFAULT_TOKEN_BUDGET

    @classmethod
    def from_settings(cls, s: Settings) -> RetrievalConfig:
        return cls(
            dense_top_k=s.retrieval_dense_top_k,
            sparse_top_k=s.retrieval_sparse_top_k,
            final_k=s.retrieval_final_k,
            dense_weight=s.retrieval_dense_weight,
            sparse_weight=s.retrieval_sparse_weight,
            rerank_top_n=s.retrieval_rerank_top_n,
            rerank_policy=s.retrieval_rerank_policy,
        )


@dataclass
class RetrievalResult:
    query: str
    scope: ScopeFilter
    query_scope: QueryScope
    bundle: EvidenceBundle
    candidates: list[dict[str, Any]]  # every fused candidate with leg ranks (trace)
    latency_ms: dict[str, float]
    versions: dict[str, Any] = field(default_factory=dict)
    guard_rejected: int = 0

    def as_trace(self) -> dict[str, Any]:
        return {
            "scope": dataclasses.asdict(self.scope),
            "query_scope": dataclasses.asdict(self.query_scope),
            "candidates": self.candidates,
            "latency_ms": self.latency_ms,
            "versions": self.versions,
            "guard_rejected": self.guard_rejected,
        }


class RetrievalService:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
        bm25_index: Bm25Index | None = None,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self.config = config or RetrievalConfig.from_settings(get_settings())
        self._bm25_index = bm25_index  # eval harnesses pass a prebuilt index

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = get_embedding_provider()
        return self._embedder

    @property
    def reranker(self) -> Reranker | None:
        if self._reranker is None and self.config.use_reranker:
            self._reranker = get_reranker()
        return self._reranker if self.config.use_reranker else None

    # ------------------------------------------------------------------ public

    def resolve_scope(
        self, query: str, base: ScopeFilter | None = None, *, today: datetime.date | None = None
    ) -> tuple[ScopeFilter, QueryScope]:
        qs = parse_query_scope(query, today=today)
        base = base or ScopeFilter()
        scope = dataclasses.replace(
            base,
            as_of=base.as_of or qs.as_of,
            regulation_keys=base.regulation_keys or tuple(qs.regulation_keys),
            include_superseded=base.include_superseded or qs.historical_hint or qs.change_hint,
        )
        return scope, qs

    def search(
        self,
        session: Session,
        query: str,
        *,
        scope: ScopeFilter | None = None,
        k: int | None = None,
        today: datetime.date | None = None,
    ) -> RetrievalResult:
        with span("retrieval.search", k=k or self.config.final_k):
            result = self._search(session, query, scope=scope, k=k, today=today)
        for stage in ("dense", "sparse", "exact", "rerank", "evidence"):
            if stage in result.latency_ms:
                metrics.STAGE_LATENCY.labels(stage=stage).observe(result.latency_ms[stage] / 1000)
        metrics.RETRIEVAL_CANDIDATES.observe(len(result.candidates))
        if not result.bundle.evidence:
            metrics.RETRIEVAL_NO_HIT.inc()
        return result

    def _search(
        self,
        session: Session,
        query: str,
        *,
        scope: ScopeFilter | None,
        k: int | None,
        today: datetime.date | None,
    ) -> RetrievalResult:
        cfg = self.config
        k = k or cfg.final_k
        timings: dict[str, float] = {}
        t_all = time.perf_counter()

        scope, qs = self.resolve_scope(query, scope, today=today)

        dense_ids: list[uuid.UUID] = []
        sparse_ids: list[uuid.UUID] = []
        exact_ids: list[uuid.UUID] = []

        if cfg.use_dense:
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            timings["embed_query"] = _ms(t0)
            t0 = time.perf_counter()
            dense_ids = [
                cid for cid, _ in dense_search(session, qvec, scope, self.embedder, top_k=cfg.dense_top_k, today=today)
            ]
            timings["dense"] = _ms(t0)
        if cfg.use_sparse:
            t0 = time.perf_counter()
            sparse_ids = [
                cid
                for cid, _ in sparse_search(
                    session, query, scope, top_k=cfg.sparse_top_k, index=self._bm25_index, today=today
                )
            ]
            timings["sparse"] = _ms(t0)
        if cfg.use_exact and qs.has_exact_identifier:
            t0 = time.perf_counter()
            exact_ids = self._exact_leg(session, qs, scope, today=today)
            timings["exact"] = _ms(t0)

        lists, weights = [], []
        for ids, w in (
            (dense_ids, self.config.dense_weight),
            (sparse_ids, self.config.sparse_weight),
            (exact_ids, EXACT_LEG_WEIGHT),
        ):
            if ids:
                lists.append(ids)
                weights.append(w)
        fused = reciprocal_rank_fusion(lists, weights=weights) if lists else {}
        fused_order = sorted(fused, key=lambda c: fused[c], reverse=True)
        rows = rows_by_id(session, scope, fused_order, today=today)
        fused_order = [c for c in fused_order if c in rows]

        dense_rank = {c: i for i, c in enumerate(dense_ids, 1)}
        sparse_rank = {c: i for i, c in enumerate(sparse_ids, 1)}
        exact_rank = {c: i for i, c in enumerate(exact_ids, 1)}

        # rerank the fused head (filters come after ranking so tight guards don't starve results)
        rerank_scores: dict[uuid.UUID, float] = {}
        rr_meta: dict[str, Any] = {}
        degraded: list[str] = []
        skip_rerank = self.config.rerank_policy == "adaptive" and (
            (qs.has_exact_identifier and bool(exact_ids))
            or (bool(dense_ids) and bool(sparse_ids) and dense_ids[0] == sparse_ids[0])
        )
        if skip_rerank:
            rr_meta = {"reranker": "skipped:legs_agree"}
            timings["rerank"] = 0.0
        if fused_order and not skip_rerank:
            t0 = time.perf_counter()
            top_n = self.config.rerank_top_n
            head, tail = (fused_order[:top_n], fused_order[top_n:]) if top_n else (fused_order, [])
            try:
                obs = apply_reranker(
                    self.reranker,
                    query,
                    [
                        RerankCandidate(
                            id=c,
                            content=rows[c].chunk.content,
                            authority_level=rows[c].regulation.authority_level,
                            fused_score=fused[c],
                            normative=rows[c].section.normative,
                            chunk_type=rows[c].chunk.chunk_type,
                        )
                        for c in head
                    ],
                )
            except Exception as exc:  # noqa: BLE001 — reranker is a quality stage, never an availability one
                log.warning("reranker failed, using fused order: %s", exc)
                degraded.append(f"reranker_failed:{type(exc).__name__}")
                obs = None
            if obs:
                rerank_scores = obs.scores
                rr_meta = {"reranker": obs.model_name, "reranker_version": obs.model_version}
                fused_order = sorted(head, key=lambda c: rerank_scores[c], reverse=True) + tail
            timings["rerank"] = _ms(t0)

        # guard + diversify. The per-version cap only makes sense when several
        # documents compete; a query scoped to one regulation may legitimately be
        # answered by many chunks of that one text.
        distinct_versions = {rows[c].version.id for c in fused_order}
        cap = k if len(distinct_versions) <= 1 else max(MAX_CHUNKS_PER_VERSION, -(-k // 2))
        selected: list[tuple[CandidateRow, LegRanks]] = []
        per_version: dict[uuid.UUID, int] = {}
        seen_sha: set[str] = set()
        rejected = 0
        trace: list[dict[str, Any]] = []
        for c in fused_order:
            row = rows[c]
            ranks = LegRanks(
                dense=dense_rank.get(c),
                sparse=sparse_rank.get(c),
                exact=exact_rank.get(c),
                fused_score=fused[c],
                rerank_score=rerank_scores.get(c),
            )
            entry = {"chunk_id": str(c), "citation": row.chunk.citation_label, **ranks.model_dump(), "selected": False}
            trace.append(entry)
            if len(selected) >= k:
                continue
            exact_hit = c in exact_rank
            if not exact_hit and not has_known_authority(row.regulation.authority_level):
                rejected += 1
                continue
            if not exact_hit and not is_relevant(query, row.chunk.content, min_shared_terms=cfg.min_shared_terms):
                rejected += 1
                entry["rejected"] = "relevance_floor"
                continue
            if row.chunk.chunk_sha256 in seen_sha or per_version.get(row.version.id, 0) >= cap:
                entry["rejected"] = "dedup"
                continue
            seen_sha.add(row.chunk.chunk_sha256)
            per_version[row.version.id] = per_version.get(row.version.id, 0) + 1
            entry["selected"] = True
            selected.append((row, ranks))

        t0 = time.perf_counter()
        bundle = build_evidence(
            session,
            selected,
            token_budget=cfg.token_budget,
            expand_parents=cfg.expand_parents,
            expand_cross_refs=cfg.expand_cross_refs,
        )
        timings["evidence"] = _ms(t0)
        timings["total"] = _ms(t_all)
        return RetrievalResult(
            query=query,
            scope=scope,
            query_scope=qs,
            bundle=bundle,
            candidates=trace,
            latency_ms=timings,
            versions={
                "embedding_model": self.embedder.model_name if cfg.use_dense else None,
                **rr_meta,
                "config": dataclasses.asdict(cfg),
                "degraded": degraded,
            },
            guard_rejected=rejected,
        )

    # ------------------------------------------------------------------ legs

    def _exact_leg(
        self, session: Session, qs: QueryScope, scope: ScopeFilter, *, today: datetime.date | None
    ) -> list[uuid.UUID]:
        """Chunks whose section path (or merged clause paths) matches a requested clause/annex."""
        conds = []
        for num in qs.clause_numbers:
            conds.append(Section.path == num)
            conds.append(Section.path.like(f"%/{num}"))
            conds.append(Chunk.metadata_["merged_paths"].as_string().like(f'%"{num}"%'))
            conds.append(Chunk.metadata_["merged_paths"].as_string().like(f'%/{num}"%'))
        if not qs.clause_numbers:
            for annex in qs.annexes:
                conds.append(Section.path.like(f"annex-{annex.lower()}/%"))
                conds.append(Section.path == f"annex-{annex.lower()}")
        if not conds:
            return []
        stmt = (
            scoped_statement(scope, today=today)
            .with_only_columns(Chunk.id)
            .where(or_(*conds))
            .order_by(Chunk.ordinal)
            .limit(20)
        )
        return list(session.scalars(stmt).all())


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def search_ids(session: Session, service: RetrievalService, query: str, **kw: Any) -> list[uuid.UUID]:
    """Convenience for evaluation: ranked chunk ids of the final evidence."""
    return [e.chunk_id for e in service.search(session, query, **kw).bundle.evidence]


__all__ = ["RetrievalConfig", "RetrievalResult", "RetrievalService", "search_ids"]
