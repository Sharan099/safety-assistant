"""Retrieval breadth metrics for eval regression gates."""

from __future__ import annotations

from typing import Any

from backend.app.retrieval.applicability_boost import detect_query_categories
from backend.app.retrieval.citations import chunk_authority_for_categories, enrich_doc_provenance


def _blob(documents: list[dict], chunk_lookup: dict | None) -> str:
    parts: list[str] = []
    for d in documents:
        text = d.get("text") or ""
        if chunk_lookup and d.get("id") in chunk_lookup:
            text = chunk_lookup[d["id"]].get("text") or text
        parts.append(text.lower())
    return " ".join(parts)


def retrieval_signal_hits(blob: str, signals: list[str]) -> dict[str, bool]:
    patterns = {
        "load": ("dan", "load", "tractive", "force"),
        "duration": ("0.2", "second", "duration", "hold"),
        "angle": ("angle", "degree", "°"),
        "geometric": ("displacement", "reference zone", "geometric", "distance", "mm"),
    }
    return {
        sig: any(p in blob for p in patterns.get(sig, (sig,)))
        for sig in signals
    }


def applicability_match_rate(
    query: str,
    documents: list[dict],
    chunk_lookup: dict | None = None,
) -> float:
    """Fraction of retrieved chunks whose applicability header matches query category."""
    q_cats = detect_query_categories(query)
    if not q_cats:
        return 1.0
    matched = 0
    scored = 0
    for d in documents:
        enriched = enrich_doc_provenance(d, chunk_lookup)
        applies = enriched.get("applies_to_category") or []
        if not applies:
            continue
        scored += 1
        if chunk_authority_for_categories(enriched, q_cats):
            matched += 1
    if scored == 0:
        return 1.0
    return matched / scored


def score_retrieval_breadth(
    case: dict[str, Any],
    documents: list[dict],
    chunk_lookup: dict | None = None,
) -> dict[str, Any]:
    """Score retrieval-only breadth metrics (independent of LLM answer quality)."""
    blob = _blob(documents, chunk_lookup)
    required = case.get("required_retrieval_signals") or []
    signal_hits = retrieval_signal_hits(blob, required) if required else {}
    min_chunks = int(case.get("min_retrieved_chunks") or 0)
    chunk_count = len(documents)
    app_rate = applicability_match_rate(case["question"], documents, chunk_lookup)

    failures: list[str] = []
    if min_chunks and chunk_count < min_chunks:
        failures.append(f"chunk_count {chunk_count} < {min_chunks}")
    for sig, hit in signal_hits.items():
        if not hit:
            failures.append(f"missing_signal:{sig}")
    if case.get("query_type") == "broad_synthesis" and chunk_count < 8:
        failures.append("broad_query_under_retrieved")

    return {
        "id": case["id"],
        "chunk_count": chunk_count,
        "applicability_match_rate": round(app_rate, 4),
        "signal_hits": signal_hits,
        "pass": len(failures) == 0,
        "failures": failures,
    }


def aggregate_retrieval_breadth_stats(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "pass_rate": 1.0, "gate_pass": True}
    passed = sum(1 for r in rows if r.get("pass"))
    return {
        "samples": len(rows),
        "pass": passed,
        "pass_rate": round(passed / len(rows), 4),
        "mean_chunk_count": round(
            sum(r.get("chunk_count", 0) for r in rows) / len(rows), 2
        ),
        "mean_applicability_match_rate": round(
            sum(r.get("applicability_match_rate", 0) for r in rows) / len(rows), 4
        ),
        "gate_pass": passed == len(rows),
    }
