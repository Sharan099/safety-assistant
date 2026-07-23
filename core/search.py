"""Hybrid retrieval + reranking + grounded Claude answer generation."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import numpy as np
from loguru import logger
from sqlalchemy import and_, func, text
from sqlalchemy.orm import Session

from app.config import settings
from core.cache import retrieval_cache
from core.citations import (
    build_citations_from_chunks,
    mark_citations_referenced_in_answer,
)
from core.embedder import Embedder, get_reranker
from core.generation.answer_formatter import format_answer_for_display
from core.generation.context_builder import (
    balance_comparison_chunks,
    dedupe_overlapping_chunks,
    prepare_llm_context,
)
from core.generation.llm_output_cleaner import clean_llm_output
from core.generation.prompt_builder import build_generation_messages
from core.grounding import verify_grounding
from core.premise_check import check_premises, format_premise_notes
from core.query_rewrite import QueryAnalysis, analyze_query, sparse_query_text
from core.regulation_detect import regulation_codes
from core.rerank_tune import boost_definition_chunks
from database.models import Chunk, Document, Regulation

ENABLE_RERANKER = settings.ENABLE_RERANKER
DEBUG_TIMING = settings.DEBUG_TIMING


def max_output_tokens_for_provider(provider: str | None) -> int:
    """Per-provider output budget — Groq free-tier Pareto default is 768."""
    return settings.MAX_OUTPUT_TOKENS


def _normalize_finish_reason(raw: str, provider: str | None) -> str:
    return (raw or "").lower()


def _output_was_truncated(routing: dict[str, Any]) -> bool:
    """True when generation hit the output token cap (incomplete Citations: section)."""
    fr = _normalize_finish_reason(
        routing.get("finish_reason") or "", routing.get("provider")
    )
    if fr in ("length", "max_tokens"):
        return True
    cap = routing.get("max_output_tokens")
    comp = routing.get("completion_tokens")
    try:
        if cap is not None and comp is not None and int(comp) >= int(cap):
            return True
    except (TypeError, ValueError):
        pass
    return False


def fusion_limit_for(
    top_k: int,
    *,
    is_definition: bool = False,
    is_comparison: bool = False,
) -> int:
    """Cap candidates entering RRF + rerank; deeper pool for definition/comparison queries."""
    base = settings.FUSION_POOL_BASE
    def_pool = settings.FUSION_POOL_DEFINITION
    cmp_pool = settings.FUSION_POOL_COMPARISON
    if is_comparison:
        return min(cmp_pool, max(top_k * 2, 12))
    if is_definition:
        return min(def_pool, top_k * 3)
    return min(base, top_k * 2)


def context_top_k_slice(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Post-rerank cut for generation + RAGAS scoring (LLM_CONTEXT_TOP_K).

    Distinct from the pre-rerank fusion pool and from DEFAULT_TOP_K (rerank depth).
    """
    k = max(1, int(settings.LLM_CONTEXT_TOP_K))
    return chunks[: min(len(chunks), k)]


def adaptive_context_top_k(analysis: QueryAnalysis) -> int:
    """Phase 5: fewer chunks for simple factual/definition queries; more for comparison."""
    base = max(1, int(settings.LLM_CONTEXT_TOP_K))
    if analysis.is_comparison:
        return max(base, min(8, base + 2))
    if analysis.is_definition:
        return max(3, min(base, 4))
    # general / single-fact: lean
    return max(3, min(base, 5))


def dedupe_overlapping_chunks(
    chunks: list[dict[str, Any]], *, min_overlap_ratio: float = 0.55
) -> list[dict[str, Any]]:
    """Re-export for tests and retrieval cache paths."""
    from core.generation.context_builder import dedupe_overlapping_chunks as _dedupe

    return _dedupe(chunks, min_overlap_ratio=min_overlap_ratio)


def _apply_context_pipeline(
    chunks: list[dict[str, Any]], analysis: QueryAnalysis
) -> tuple[str, list[dict[str, Any]], int]:
    """Adaptive top-k, dedupe, balance, and grouped context string."""
    if settings.ENABLE_ADAPTIVE_CONTEXT:
        k = adaptive_context_top_k(analysis)
        sliced = chunks[: min(len(chunks), k)]
    else:
        sliced = context_top_k_slice(chunks)
        k = len(sliced)
    context_str, enriched = prepare_llm_context(
        sliced, analysis, max_chunks=k, dedupe=settings.ENABLE_CONTEXT_DEDUPE
    )
    return context_str, enriched, k


class SearchEngine:
    def __init__(self):
        self.embedder = Embedder()
        self._reranker = None
        self._reranker_load_attempted = False
        self._reranker_disabled = not ENABLE_RERANKER

    @property
    def reranker(self):
        """Lazy-load reranker so embedder-only startup succeeds when reranker weights are missing."""
        if self._reranker_disabled:
            return None
        if not self._reranker_load_attempted:
            self._reranker_load_attempted = True
            try:
                self._reranker = get_reranker()
            except Exception as exc:
                logger.warning(
                    "Reranker {} unavailable — continuing with fusion-only ranking: {}",
                    os.getenv("RERANKER_MODEL", "unknown"),
                    exc,
                )
                self._reranker_disabled = True
        return self._reranker

    def retrieve(self, db: Session, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        """Retrieval only — no LLM. Used by /search and as the first stage of /chat."""
        top_k = top_k or settings.DEFAULT_TOP_K
        start = time.perf_counter()
        analysis = analyze_query(query)
        filters = self._parse_filters(query, analysis)

        cached = retrieval_cache.get(query, top_k, filters)
        if cached is not None:
            context_str, scored, k = _apply_context_pipeline(cached, analysis)
            total_ms = (time.perf_counter() - start) * 1000
            return self._retrieval_payload(
                query, scored, context_str, analysis, filters, [], total_ms, cached=True
            )

        final, timing_parts, premise_issues = self._run_retrieval(
            db, query, top_k=top_k, analysis=analysis, filters=filters
        )
        retrieval_cache.put(query, top_k, filters, final)
        context_str, scored, k = _apply_context_pipeline(final, analysis)
        total_ms = (time.perf_counter() - start) * 1000
        timing_parts["total_ms"] = total_ms
        timing_parts["context_top_k"] = k
        timing_parts["context_after_dedupe"] = len(scored)
        return self._retrieval_payload(
            query, scored, context_str, analysis, filters, premise_issues, total_ms, timing_parts
        )

    def search(self, db: Session, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        top_k = top_k or settings.DEFAULT_TOP_K
        start = time.perf_counter()
        retrieved = self.retrieve(db, query, top_k=top_k)
        # retrieve() already applies LLM_CONTEXT_TOP_K to sources/context.
        llm_chunks = retrieved["sources"]
        context_str = retrieved.get("context") or ""
        premise_issues = retrieved.get("metadata", {}).get("premise_corrections", [])
        premise_notes = format_premise_notes(
            check_premises(query)
        ) if not premise_issues else "\n".join(
            ["Premise check:"] + [f"- {m}" for m in premise_issues]
        )
        analysis = analyze_query(query)

        t_llm = time.perf_counter()
        answer, routing = self._generate_answer(
            query,
            context_str,
            chunks=llm_chunks,
            premise_notes=premise_notes,
            analysis=analysis,
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000

        citations = mark_citations_referenced_in_answer(
            build_citations_from_chunks(llm_chunks), answer
        )
        grounding = verify_grounding(answer, citations)
        _output_was_truncated(routing)
        # Keep model answer — hard refusal destroyed RAGAS relevancy/correctness on
        # borderline grounding. Truncated answers skip unsupported-sentence penalties.
        if not llm_chunks:
            answer = "I could not find relevant passages in the UNECE corpus for that question."
        else:
            answer = format_answer_for_display(grounding, answer)

        total_ms = (time.perf_counter() - start) * 1000
        timing = retrieved.get("timing") or {"total_ms": total_ms}
        timing["llm_ms"] = round(llm_ms, 2)
        timing["total_ms"] = total_ms

        return {
            "query": query,
            "answer": answer,
            "sources": llm_chunks,
            "citations": citations,
            "grounding": grounding.to_dict(),
            "metadata": {
                **retrieved.get("metadata", {}),
                "routing": routing,
                "latency_ms": total_ms,
            },
            "timing": timing,
        }

    def _retrieval_payload(
        self,
        query: str,
        final: list[dict[str, Any]],
        context_str: str,
        analysis: QueryAnalysis,
        filters: dict[str, Any],
        premise_issues: list,
        total_ms: float,
        timing_parts: dict[str, Any] | None = None,
        *,
        cached: bool = False,
    ) -> dict[str, Any]:
        timing: dict[str, Any] = {"total_ms": total_ms, "cache_hit": cached}
        if timing_parts:
            timing.update(timing_parts)
        corrections = [i.message for i in premise_issues] if premise_issues else []
        if not corrections:
            corrections = [i.message for i in check_premises(query)]
        return {
            "query": query,
            "context": context_str,
            "sources": final,
            "citations": build_citations_from_chunks(final),
            "metadata": {
                "filters_applied": filters,
                "query_type": self._query_type(analysis),
                "premise_corrections": corrections,
            },
            "timing": timing,
        }

    def _run_retrieval(
        self,
        db: Session,
        query: str,
        *,
        top_k: int,
        analysis: QueryAnalysis,
        filters: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, float], list]:
        retrieval_query = self._expand_query(query, filters, analysis)
        sparse_query = sparse_query_text(analysis)
        fusion_limit = fusion_limit_for(
            top_k,
            is_definition=analysis.is_definition,
            is_comparison=analysis.is_comparison,
        )
        premise_issues = check_premises(query)

        codes = regulation_codes(query)
        if settings.ENABLE_QUERY_DECOMPOSITION and len(codes) >= 2:
            return self._retrieve_decomposed(
                db,
                query,
                top_k=top_k,
                analysis=analysis,
                codes=codes,
                premise_issues=premise_issues,
            )

        t_embed = time.perf_counter()
        query_variants = self._query_variants(query, analysis, filters)
        dense_lists: list[list[dict[str, Any]]] = []
        for variant in query_variants:
            vec = self.embedder.embed_query(variant)
            dense_lists.append(self._dense_search(db, vec, filters, fusion_limit))
        embed_ms = (time.perf_counter() - t_embed) * 1000

        t_dense = time.perf_counter()
        dense = dense_lists[0]
        for extra in dense_lists[1:]:
            dense = self._rrf_many([dense, extra], fusion_limit)
        dense_ms = (time.perf_counter() - t_dense) * 1000

        t_sparse = time.perf_counter()
        sparse = self._sparse_search(db, sparse_query, filters, fusion_limit)
        sparse_ms = (time.perf_counter() - t_sparse) * 1000

        t_rrf = time.perf_counter()
        fused = self._rrf(dense, sparse, fusion_limit)
        rrf_ms = (time.perf_counter() - t_rrf) * 1000

        fused, rerank_ms = self._rerank_fused(query, fused)
        fused = boost_definition_chunks(fused, enabled=analysis.is_definition)
        final = self._promote_sections(query, fused, top_k)[:top_k]
        if analysis.is_definition:
            final = self._promote_definition_hits(analysis, final, top_k)

        timing = {
            "embed_ms": round(embed_ms, 2),
            "dense_ms": round(dense_ms, 2),
            "sparse_ms": round(sparse_ms, 2),
            "rrf_ms": round(rrf_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "fusion_limit": fusion_limit,
        }
        if DEBUG_TIMING:
            timing["query_type"] = self._query_type(analysis)
            timing["premise_corrections"] = len(premise_issues)
        return final, timing, premise_issues

    def _rerank_fused(
        self, query: str, fused: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], float]:
        t_rerank = time.perf_counter()
        if self.reranker is not None:
            passages = [c["chunk_text"] for c in fused]
            scores = self.reranker.score(query, passages)
            for idx, score in enumerate(scores):
                fused[idx]["score"] = score
            fused.sort(key=lambda x: x["score"], reverse=True)
        else:
            for item in fused:
                item["score"] = item.get("rrf_score", item.get("search_score", 0.0))
            fused.sort(key=lambda x: x["score"], reverse=True)
        return fused, (time.perf_counter() - t_rerank) * 1000

    def _retrieve_decomposed(
        self,
        db: Session,
        query: str,
        *,
        top_k: int,
        analysis: QueryAnalysis,
        codes: list[str],
        premise_issues: list,
    ) -> tuple[list[dict[str, Any]], dict[str, float], list]:
        """Retrieve per named regulation, RRF-merge lists, then rerank on the full query."""
        fusion_limit = fusion_limit_for(
            top_k,
            is_definition=analysis.is_definition,
            is_comparison=True,
        )
        sparse_query = sparse_query_text(analysis)
        per_reg_lists: list[list[dict[str, Any]]] = []
        embed_ms = dense_ms = sparse_ms = 0.0

        for code in codes:
            sub_filters = {"regulation_code": code}
            retrieval_query = self._expand_query(query, sub_filters, analysis)
            t_embed = time.perf_counter()
            vec = self.embedder.embed_query(retrieval_query)
            embed_ms += (time.perf_counter() - t_embed) * 1000
            t_dense = time.perf_counter()
            dense = self._dense_search(db, vec, sub_filters, fusion_limit)
            dense_ms += (time.perf_counter() - t_dense) * 1000
            t_sparse = time.perf_counter()
            sparse = self._sparse_search(db, sparse_query, sub_filters, fusion_limit)
            sparse_ms += (time.perf_counter() - t_sparse) * 1000
            per_reg_lists.append(self._rrf(dense, sparse, fusion_limit))

        t_rrf = time.perf_counter()
        fused = self._rrf_many(per_reg_lists, fusion_limit)
        rrf_ms = (time.perf_counter() - t_rrf) * 1000

        fused, rerank_ms = self._rerank_fused(query, fused)
        fused = boost_definition_chunks(fused, enabled=analysis.is_definition)
        # Global rerank favors one regulation on comparison queries — reserve slots per code.
        if len(codes) >= 2:
            fused = balance_comparison_chunks(fused, codes, max_total=top_k)
        final = self._promote_sections(query, fused, top_k)[:top_k]
        if analysis.is_definition:
            final = self._promote_definition_hits(analysis, final, top_k)

        timing = {
            "embed_ms": round(embed_ms, 2),
            "dense_ms": round(dense_ms, 2),
            "sparse_ms": round(sparse_ms, 2),
            "rrf_ms": round(rrf_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "fusion_limit": fusion_limit,
            "decomposed_regs": len(codes),
        }
        if DEBUG_TIMING:
            timing["query_type"] = self._query_type(analysis)
            timing["premise_corrections"] = len(premise_issues)
        return final, timing, premise_issues

    def _query_variants(
        self, query: str, analysis: QueryAnalysis, filters: dict[str, Any]
    ) -> list[str]:
        variants = [self._expand_query(query, filters, analysis)]
        if settings.ENABLE_MULTI_QUERY and analysis.retrieval_query != query:
            variants.append(query)
        return variants[: max(1, settings.MULTI_QUERY_COUNT)]

    def _rrf_many(self, lists: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
        k = 60
        merged: dict[int, dict[str, Any]] = {}
        for lst in lists:
            for rank, item in enumerate(lst):
                cid = item["chunk_id"]
                score = 1.0 / (k + rank + 1)
                if cid in merged:
                    merged[cid]["score"] += score
                else:
                    merged[cid] = {"item": item, "score": score}
        ordered = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:limit]
        out = []
        for row in ordered:
            item = row["item"]
            item["rrf_score"] = row["score"]
            out.append(item)
        return out

    def _query_type(self, analysis: QueryAnalysis) -> str:
        if analysis.is_comparison:
            return "comparison"
        if analysis.is_definition:
            return "definition"
        return "general"

    def _promote_definition_hits(
        self, analysis: QueryAnalysis, ranked: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Ensure definition-section chunks mentioning the term appear in the LLM context."""
        term = (analysis.defined_term or "").lower()
        if not term or not ranked:
            return ranked

        term_re = re.compile(rf"\b{re.escape(term)}\b", re.I)
        hits = [
            c
            for c in ranked
            if term_re.search(c.get("chunk_text") or "")
            or term_re.search(c.get("section") or "")
        ]
        if not hits:
            return ranked

        best = max(hits, key=lambda c: c.get("score", c.get("rrf_score", 0.0)))
        top = ranked[:top_k]
        if any(c["chunk_id"] == best["chunk_id"] for c in top):
            return ranked
        return ranked[: top_k - 1] + [best]

    def _parse_filters(self, query: str, analysis: QueryAnalysis | None = None) -> dict[str, Any]:
        """Pre-filter retrieval to named regulation(s).

        Comparison queries with multiple explicit regulations use an OR filter
        (regulation_codes) so hybrid search cannot collapse to a single reg.
        """
        analysis = analysis or analyze_query(query)
        filters: dict[str, Any] = {}
        codes = regulation_codes(query)

        # Multi-regulation queries: OR-filter so retrieval cannot collapse to one reg.
        if len(codes) >= 2:
            filters["regulation_codes"] = codes
            return filters

        if not settings.ENABLE_REGULATION_PREFILTER:
            return filters

        if not codes:
            return filters
        if len(codes) == 1:
            filters["regulation_code"] = codes[0]
        else:
            filters["regulation_codes"] = codes
        return filters

    def _expand_query(self, query: str, filters: dict[str, Any], analysis: QueryAnalysis) -> str:
        base = analysis.retrieval_query
        q = query.lower()
        extras: list[str] = []
        reg = filters.get("regulation_code", "")
        if reg == "UN_R94" or re.search(r"\bR\s*94\b", query, re.I):
            if any(w in q for w in ("chest", "thorax", "thcc", "hic", "injury")):
                extras.extend(["ThCC", "HIC", "5.2.1.4", "injury criteria"])
        if reg == "UN_R16" or re.search(r"\bR\s*16\b", query, re.I):
            if any(w in q for w in ("belt", "retractor", "lock", "anchorage")):
                extras.extend(["safety belt", "retractor", "anchorage", "7.6.2"])
        if reg == "UN_R129" or re.search(r"\bR\s*129\b", query, re.I):
            extras.extend(["i-Size", "child restraint", "ISOFIX", "height class"])
        if analysis.is_definition and analysis.defined_term:
            extras.extend(["definitions", "means", "shall mean", analysis.defined_term])
        if extras:
            return f"{base} {' '.join(extras)}"
        return base

    def _filter_conditions(self, filters: dict[str, Any]) -> list:
        cond = []
        if filters.get("regulation_codes"):
            cond.append(Regulation.regulation_code.in_(filters["regulation_codes"]))
        elif filters.get("regulation_code"):
            cond.append(Regulation.regulation_code == filters["regulation_code"])
        return cond

    def _dense_search(
        self, db: Session, embedding: list[float], filters: dict[str, Any], limit: int
    ) -> list[dict[str, Any]]:
        if db.bind.dialect.name == "sqlite":
            return self._dense_in_memory(db, embedding, filters, limit)
        try:
            return self._dense_pgvector(db, embedding, filters, limit)
        except Exception as exc:
            logger.warning("pgvector search failed (%s); using in-memory fallback", exc)
            return self._dense_in_memory(db, embedding, filters, limit)

    def _vec_literal(self, embedding: list[float]) -> str:
        return "[" + ",".join(f"{float(x):.8g}" for x in embedding) + "]"

    def _dense_pgvector(
        self, db: Session, embedding: list[float], filters: dict[str, Any], limit: int
    ) -> list[dict[str, Any]]:
        vec = self._vec_literal(embedding)
        sql = """
            SELECT chunks.id,
                   1 - (chunks.embedding <=> CAST(:qvec AS vector)) AS score
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            JOIN regulations ON regulations.id = documents.regulation_id
            WHERE chunks.embedding IS NOT NULL
        """
        params: dict[str, Any] = {"qvec": vec, "lim": limit}
        if filters.get("regulation_codes"):
            sql += " AND regulations.regulation_code = ANY(:codes)"
            params["codes"] = filters["regulation_codes"]
        elif filters.get("regulation_code"):
            sql += " AND regulations.regulation_code = :reg_code"
            params["reg_code"] = filters["regulation_code"]
        sql += " ORDER BY chunks.embedding <=> CAST(:qvec AS vector) LIMIT :lim"
        rows = db.execute(text(sql), params).all()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        score_map = {int(r[0]): float(r[1]) for r in rows}
        chunks = db.query(Chunk).filter(Chunk.id.in_(ids)).all()
        chunks.sort(key=lambda c: score_map.get(c.id, 0), reverse=True)
        return [self._format(c, score_map[c.id]) for c in chunks]

    def _dense_in_memory(
        self, db: Session, embedding: list[float], filters: dict[str, Any], limit: int
    ) -> list[dict[str, Any]]:
        stmt = db.query(Chunk).join(Document).join(Regulation)
        cond = self._filter_conditions(filters)
        if cond:
            stmt = stmt.filter(and_(*cond))
        q = np.array(embedding, dtype=float)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        scored: list[tuple[Any, float]] = []
        for chunk in stmt.all():
            if not chunk.embedding:
                continue
            raw = chunk.embedding
            if isinstance(raw, str):
                raw = json.loads(raw)
            v = np.array(raw, dtype=float)
            vn = np.linalg.norm(v)
            if vn == 0:
                continue
            scored.append((chunk, float(np.dot(q, v) / (qn * vn))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._format(c, s) for c, s in scored[:limit]]

    def _sparse_search(
        self, db: Session, query: str, filters: dict[str, Any], limit: int
    ) -> list[dict[str, Any]]:
        if db.bind.dialect.name == "sqlite":
            return self._sparse_like(db, query, filters, limit)
        stmt = db.query(
            Chunk,
            func.ts_rank(
                func.to_tsvector("english", Chunk.chunk_text),
                func.plainto_tsquery("english", query),
            ).label("score"),
        ).join(Document).join(Regulation)
        cond = [
            func.plainto_tsquery("english", query).op("@@")(
                func.to_tsvector("english", Chunk.chunk_text)
            )
        ]
        cond.extend(self._filter_conditions(filters))
        rows = stmt.filter(and_(*cond)).order_by(text("score DESC")).limit(limit).all()
        return [self._format(chunk, float(score)) for chunk, score in rows]

    def _sparse_like(
        self, db: Session, query: str, filters: dict[str, Any], limit: int
    ) -> list[dict[str, Any]]:
        terms = [t.lower() for t in re.sub(r"[^\w\s]", "", query).split() if t]
        if not terms:
            return []
        stmt = db.query(Chunk).join(Document).join(Regulation)
        cond = self._filter_conditions(filters)
        if cond:
            stmt = stmt.filter(and_(*cond))
        scored: list[tuple[Any, float]] = []
        for chunk in stmt.all():
            text_l = (chunk.chunk_text or "").lower()
            score = sum(1 for t in terms if t in text_l)
            if score:
                scored.append((chunk, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._format(c, s) for c, s in scored[:limit]]

    def _rrf(
        self, dense: list[dict[str, Any]], sparse: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        k = 60
        merged: dict[int, dict[str, Any]] = {}
        for rank, item in enumerate(dense):
            cid = item["chunk_id"]
            merged[cid] = {"item": item, "score": 1.0 / (k + rank + 1)}
        for rank, item in enumerate(sparse):
            cid = item["chunk_id"]
            if cid in merged:
                merged[cid]["score"] += 1.0 / (k + rank + 1)
            else:
                merged[cid] = {"item": item, "score": 1.0 / (k + rank + 1)}
        ordered = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:limit]
        out = []
        for row in ordered:
            item = row["item"]
            item["rrf_score"] = row["score"]
            out.append(item)
        return out

    def _promote_sections(
        self, query: str, fused: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        cited = set(re.findall(r"(?:§|paragraph\s+)(\d+(?:\.\d+)*)", query, re.I))
        cited.update(re.findall(r"\b(\d+\.\d+(?:\.\d+)*)\b", query))
        if not cited:
            return fused
        hits = [
            c
            for c in fused
            if any(
                (c.get("section") or "") == s
                or (c.get("section") or "").startswith(s + ".")
                for s in cited
            )
        ]
        if not hits:
            return fused
        best = max(hits, key=lambda c: c.get("score", c.get("rrf_score", 0)))
        top = fused[:top_k]
        if any(c["chunk_id"] == best["chunk_id"] for c in top):
            return top
        return top[: top_k - 1] + [best]

    def _format(self, chunk: Chunk, score: float) -> dict[str, Any]:
        return {
            "chunk_id": chunk.id,
            "chunk_text": chunk.chunk_text,
            "page_number": chunk.page_number,
            "section": chunk.section,
            "document_id": chunk.document_id,
            "document_name": chunk.document.document_name,
            "regulation_code": chunk.document.regulation.regulation_code,
            "title": chunk.document.regulation.title,
            "amendment": chunk.document.regulation.amendment,
            "source_type": chunk.document.regulation.source_type,
            "search_score": float(score),
        }

    def _generate_answer(
        self,
        query: str,
        context: str,
        *,
        chunks: list[dict[str, Any]] | None = None,
        premise_notes: str = "",
        analysis: QueryAnalysis | None = None,
    ) -> tuple[str, dict[str, Any]]:
        from backend.app.gateway import config as gateway_config
        from backend.app.gateway.gateway import LLMGateway

        if not context.strip():
            routing = {
                "model_key": "none",
                "model_id": "none",
                "provider": "registry",
                "evidence_only": True,
                "latency_ms": 0.0,
            }
            return (
                "I could not find relevant passages in the UNECE corpus for that question.",
                routing,
            )

        analysis = analysis or analyze_query(query)
        messages = build_generation_messages(
            query,
            context,
            premise_notes=premise_notes,
            analysis=analysis,
        )

        if gateway_config.ENABLE_GATEWAY:
            from backend.app.gateway.model_registry import get_spec, ordered_keys

            primary = gateway_config.DEFAULT_PRIMARY
            chain = ordered_keys(primary) or [primary]
            first_spec = get_spec(chain[0]) if chain else None
            provider = first_spec.provider if first_spec else "groq"
            max_out = max_output_tokens_for_provider(provider)
            result = LLMGateway().complete(
                messages,
                max_output_tokens=max_out,
                temperature=0.0,
                context_chunks=chunks or [],
            )
            routing = result.to_dict()
            routing["max_output_tokens"] = max_out
            routing["temperature"] = 0.0
            return clean_llm_output(result.text), routing

        return "Evidence-only mode: see cited passages in Context.", {
            "model_key": "evidence_only",
            "model_id": "none",
            "provider": "registry",
            "evidence_only": True,
            "latency_ms": 0.0,
        }

    def _evidence_summary(self, chunks: list[dict[str, Any]]) -> str:
        lines = ["Relevant passages (evidence-only mode):"]
        for idx, chunk in enumerate(chunks[:5], start=1):
            snippet = (chunk["chunk_text"] or "")[:400]
            lines.append(
                f"[S{idx}] {chunk['regulation_code']} p.{chunk['page_number']}: {snippet}..."
            )
        return "\n".join(lines)
