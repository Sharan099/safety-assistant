"""Context assembly for LLM generation — dedupe, balance, sort."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from core.citations import build_grouped_context
from core.query_rewrite import QueryAnalysis


def dedupe_overlapping_chunks(
    chunks: list[dict[str, Any]], *, min_overlap_ratio: float = 0.55
) -> list[dict[str, Any]]:
    """Drop later chunks whose text heavily overlaps an earlier kept chunk."""
    if len(chunks) <= 1:
        return chunks
    kept: list[dict[str, Any]] = []
    kept_norms: list[set[str]] = []
    for chunk in chunks:
        text = re.sub(r"\s+", " ", (chunk.get("chunk_text") or "")).lower()
        toks = {t for t in text.split() if len(t) > 3}
        if not toks:
            kept.append(chunk)
            kept_norms.append(toks)
            continue
        redundant = False
        for prior in kept_norms:
            if not prior:
                continue
            inter = len(toks & prior)
            ratio = inter / max(1, min(len(toks), len(prior)))
            if ratio >= min_overlap_ratio:
                redundant = True
                break
        if not redundant:
            kept.append(chunk)
            kept_norms.append(toks)
    return kept


def _sort_by_relevance(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        chunks,
        key=lambda c: float(c.get("score") or c.get("rrf_score") or c.get("search_score") or 0.0),
        reverse=True,
    )


def balance_comparison_chunks(
    chunks: list[dict[str, Any]],
    regulation_codes: list[str],
    *,
    max_total: int,
    min_per_reg: int = 2,
) -> list[dict[str, Any]]:
    """Ensure multi-regulation comparison context includes chunks from each named regulation."""
    if len(regulation_codes) < 2 or not chunks:
        return chunks[:max_total]

    by_reg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in _sort_by_relevance(chunks):
        reg = chunk.get("regulation_code") or "UNKNOWN"
        by_reg[reg].append(chunk)

    selected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    # Reserve slots per mentioned regulation (when chunks exist).
    for code in regulation_codes:
        pool = by_reg.get(code, [])
        take = min(min_per_reg, len(pool), max(1, max_total // len(regulation_codes)))
        for chunk in pool[:take]:
            cid = chunk.get("chunk_id")
            if cid in seen_ids:
                continue
            selected.append(chunk)
            seen_ids.add(cid)

    # Fill remaining budget by global relevance.
    for chunk in _sort_by_relevance(chunks):
        if len(selected) >= max_total:
            break
        cid = chunk.get("chunk_id")
        if cid in seen_ids:
            continue
        selected.append(chunk)
        seen_ids.add(cid)

    return _sort_by_relevance(selected)


def prepare_llm_context(
    chunks: list[dict[str, Any]],
    analysis: QueryAnalysis,
    *,
    max_chunks: int,
    dedupe: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Sort, dedupe, optionally balance by regulation, then build grouped context."""
    if not chunks:
        return "", []

    working = _sort_by_relevance(chunks)
    if dedupe:
        working = dedupe_overlapping_chunks(working)
    if analysis.is_comparison and len(analysis.regulation_codes) >= 2:
        working = balance_comparison_chunks(
            working,
            analysis.regulation_codes,
            max_total=max_chunks,
        )
    else:
        working = working[:max_chunks]

    return build_grouped_context(working)
