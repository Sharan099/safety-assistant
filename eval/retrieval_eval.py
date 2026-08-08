"""Retrieval-only evaluation against golden_set.jsonl (no LLM calls)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from eval.gold import (
    DEFAULT_GOLDEN,
    gold_keys,
    is_hit,
    load_golden_set,
    retrieved_ranked_keys,
)
from eval.metrics import mean, mrr, ndcg_at_k, precision_at_k, recall_at_k
from retrieval.enumerative import (
    bias_chunks_for_enumerative_topic,
    classify_enumerative,
    is_enumerative_query,
)
from retrieval.retrieve import (
    RetrievedChunk,
    fetch_scope_chunks,
    hybrid_search,
    is_scope_objective_query,
    prepend_scope_chunks,
    retrieve,
)

logger = logging.getLogger(__name__)

RetrieveFn = Callable[..., list[RetrievedChunk]]

# Scorecard-required retrieval metrics.
SCORECARD_K = {
    "recall@5": ("recall", 5),
    "precision@5": ("precision", 5),
    "ndcg@10": ("ndcg", 10),
}

# Categories with a meaningful expected_chunk_ids answer key — the only ones
# that belong in MRR / recall / precision aggregates.
CHUNK_GOLD_CATEGORIES = frozenset(
    {
        "factual_lookup",
        "compliance_check",
        "multi_hop",
        "enumerative",
        "cross_regulation",
        "design_implication",
    }
)
# No single "correct chunk" concept — never blend into retrieval quality.
NON_RETRIEVAL_CATEGORIES = frozenset(
    {
        "numeric_safety",
        "guardrail",
        "prompt_injection",
        "hallucination_probe",
        "out_of_scope",
    }
)


def _metric_averages(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    if not rows:
        return {
            "n_cases": 0,
            "recall@5": None,
            "precision@5": None,
            "mrr": None,
            "ndcg@10": None,
        }
    out: dict[str, Any] = {
        "n_cases": len(rows),
        "recall@5": round(mean([c.get("recall@5", 0.0) for c in rows]), 4),
        "precision@5": round(mean([c.get("precision@5", 0.0) for c in rows]), 4),
        "mrr": round(mean([c.get("mrr", 0.0) for c in rows]), 4),
        "ndcg@10": round(mean([c.get("ndcg@10", 0.0) for c in rows]), 4),
    }
    for k in k_values:
        out[f"recall@{k}"] = round(mean([c.get(f"recall@{k}", 0.0) for c in rows]), 4)
        out[f"precision@{k}"] = round(
            mean([c.get(f"precision@{k}", 0.0) for c in rows]), 4
        )
        out[f"ndcg@{k}"] = round(mean([c.get(f"ndcg@{k}", 0.0) for c in rows]), 4)
    return out


def compute_retrieval_quality_report(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Split MRR/recall/precision into chunk-gold vs excluded categories.

    Works on retrieval-eval rows (``retrieved_chunk_ids`` / ``mrr``) or full
    ``results.json`` case rows (``retrieved_chunks`` + ``expected_chunk_ids``).
    """
    from eval.metrics import mrr as mrr_fn
    from eval.metrics import precision_at_k, recall_at_k

    def _row_metrics(case: dict[str, Any]) -> dict[str, float] | None:
        if "mrr" in case and case.get("mrr") is not None:
            return {
                "mrr": float(case.get("mrr") or 0.0),
                "recall@5": float(case.get("recall@5") or 0.0),
                "precision@5": float(case.get("precision@5") or 0.0),
                "ndcg@10": float(case.get("ndcg@10") or 0.0),
            }
        expected = [
            str(x).strip()
            for x in (case.get("expected_chunk_ids") or [])
            if str(x).strip()
        ]
        if not expected:
            return None
        retrieved: list[str] = []
        if case.get("retrieved_chunk_ids"):
            retrieved = [
                str(x).strip() for x in case["retrieved_chunk_ids"] if str(x).strip()
            ]
        else:
            for ch in case.get("retrieved_chunks") or []:
                if isinstance(ch, dict):
                    cid = str(ch.get("chunk_id") or "").strip()
                else:
                    cid = str(getattr(ch, "chunk_id", "") or "").strip()
                if cid:
                    retrieved.append(cid)
        if not retrieved:
            return None
        return {
            "mrr": mrr_fn(retrieved, expected),
            "recall@5": recall_at_k(retrieved, expected, 5),
            "precision@5": precision_at_k(retrieved, expected, 5),
            "ndcg@10": 0.0,
        }

    chunk_gold: list[dict[str, float]] = []
    excluded: list[dict[str, float]] = []
    for case in cases:
        cat = str(case.get("category") or "").strip().lower()
        metrics = _row_metrics(case)
        if metrics is None:
            continue
        if cat in CHUNK_GOLD_CATEGORIES:
            chunk_gold.append(metrics)
        elif cat in NON_RETRIEVAL_CATEGORIES:
            excluded.append(metrics)

    def _avg(rows: list[dict[str, float]], key: str) -> float | None:
        if not rows:
            return None
        return round(sum(r[key] for r in rows) / len(rows), 4)

    return {
        "chunk_gold": {
            "label": "Retrieval quality (chunk-gold categories only)",
            "n_cases": len(chunk_gold),
            "categories": sorted(CHUNK_GOLD_CATEGORIES),
            "mrr": _avg(chunk_gold, "mrr"),
            "recall@5": _avg(chunk_gold, "recall@5"),
            "precision@5": _avg(chunk_gold, "precision@5"),
        },
        "excluded_non_retrieval": {
            "label": "Excluded from retrieval quality (non-retrieval categories)",
            "n_cases": len(excluded),
            "categories": sorted(NON_RETRIEVAL_CATEGORIES),
            "mrr": _avg(excluded, "mrr"),
            "recall@5": _avg(excluded, "recall@5"),
            "precision@5": _avg(excluded, "precision@5"),
            "note": "Not blended into chunk-gold retrieval score",
        },
    }


def format_retrieval_quality_footer(report: dict[str, Any]) -> str:
    cg = report.get("chunk_gold") or {}
    ex = report.get("excluded_non_retrieval") or {}
    cg_mrr = cg.get("mrr")
    cg_r = cg.get("recall@5")
    return (
        f"Retrieval quality (chunk-gold only, n={cg.get('n_cases', 0)}): "
        f"MRR={cg_mrr if cg_mrr is not None else '—'}  "
        f"recall@5={cg_r if cg_r is not None else '—'}  "
        f"|  excluded non-retrieval cats n={ex.get('n_cases', 0)} "
        f"(not blended)"
    )


def _is_enumerative_case(case: dict[str, Any], query: str) -> bool:
    cat = str(case.get("category") or "").strip().lower()
    if cat == "enumerative":
        return True
    return is_enumerative_query(query)


def run_retrieval_eval(
    *,
    gold_path: Path | None = None,
    k_values: list[int] | None = None,
    use_full_pipeline: bool = False,
    profile: str = "hybrid",
    retrieve_fn: RetrieveFn | None = None,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compute recall@5, precision@5, MRR, NDCG@10 (+ optional extra k).

    Profiles (for baseline vs hybrid vs +rerank comparisons):
      - dense / hybrid: hybrid_search only (no rewrite LLM)
      - rerank / pipeline: retrieve(rewrite=False, rerank=True, small_to_big=True)

    Enumerative queries on the hybrid profile get the same topic-bias + broader
    hybrid_k the live retrieve() path uses — otherwise Stage 5 under-reports
    list/every coverage (preamble crowds out door clauses, etc.).

    For ``category=enumerative``, the primary success signal is ``recall@5``
    (fraction of expected IDs in top-k), not MRR.
    """
    cases = load_golden_set(gold_path or DEFAULT_GOLDEN)
    if case_ids:
        want = {str(x).strip() for x in case_ids if str(x).strip()}
        cases = [c for c in cases if str(c.get("id") or "") in want]
    # Always include scorecard ks.
    k_values = sorted(set(k_values or [5, 10]) | {5, 10})
    max_k = max(k_values)

    profile = (profile or "hybrid").lower()
    if profile in {"rerank", "pipeline", "full"}:
        use_full_pipeline = True

    per_case: list[dict[str, Any]] = []

    for case in cases:
        query = case["question"]
        reg = case.get("regulation_id")
        gold = gold_keys(case)
        enumerative = _is_enumerative_case(case, query)
        enum_cls = classify_enumerative(query) if enumerative else None

        if retrieve_fn is not None:
            chunks = retrieve_fn(query, regulation_id=reg)
        elif use_full_pipeline:
            top = max(10, max_k)
            if enum_cls is not None and enum_cls.is_enumerative:
                top = max(top, enum_cls.rerank_top_k)
            chunks = retrieve(
                query,
                top_k=top,
                regulation_id=reg,
                rewrite=False,
                do_rerank=True,
                small_to_big=True,
            )
        else:
            hybrid_k = max(30, max_k)
            if enum_cls is not None and enum_cls.is_enumerative:
                hybrid_k = max(hybrid_k, enum_cls.hybrid_top_k)
            chunks = hybrid_search(
                query,
                top_k=hybrid_k,
                regulation_id=reg,
            )
            if enumerative:
                chunks = bias_chunks_for_enumerative_topic(chunks, question=query)
            else:
                try:
                    from retrieval.value_limit import (
                        bias_chunks_for_value_vs_limit,
                        is_compliance_prefer_limit_query,
                    )

                    if is_compliance_prefer_limit_query(query):
                        chunks = bias_chunks_for_value_vs_limit(
                            chunks, question=query
                        )
                except Exception:  # noqa: BLE001
                    pass
            if is_scope_objective_query(query):
                scope = fetch_scope_chunks(regulation_id=reg)
                scope_ids = {c.chunk_id for c in scope if c.chunk_id}
                hybrid_tail = [c for c in chunks if c.chunk_id not in scope_ids][:1]
                chunks = prepend_scope_chunks(scope, hybrid_tail)

        ranked = retrieved_ranked_keys(chunks)
        hits = [c.chunk_id for c in chunks if is_hit(c, gold)]

        # Chunk-id-only view (enumerative gold lists many valid IDs — MRR/recall@5
        # on expanded section keys understates coverage).
        expected_cids = [
            str(x).strip()
            for x in (case.get("expected_chunk_ids") or [])
            if str(x).strip()
        ]
        top_chunk_ids = [c.chunk_id for c in chunks if c.chunk_id][:20]
        chunk_hits_at_5 = [cid for cid in expected_cids if cid in set(top_chunk_ids[:5])]
        chunk_hits_at_20 = [
            cid for cid in expected_cids if cid in set(top_chunk_ids[:20])
        ]
        chunk_recall_5 = (
            len(chunk_hits_at_5) / len(expected_cids) if expected_cids else 0.0
        )
        chunk_recall_20 = (
            len(chunk_hits_at_20) / len(expected_cids) if expected_cids else 0.0
        )

        # Enumerative primary = coverage of expected chunk IDs within enum breadth.
        primary = "chunk_recall@20" if enumerative else "mrr"
        row: dict[str, Any] = {
            "id": case["id"],
            "category": case.get("category"),
            "question": query,
            "gold": sorted(gold),
            "expected_chunk_ids": expected_cids,
            "retrieved_keys": ranked[:max_k],
            "retrieved_chunk_ids": top_chunk_ids[:20],
            "hit_chunk_ids": hits,
            "n_retrieved": len(chunks),
            "abstention": case.get("abstention", False),
            "enumerative": enumerative,
            "primary_metric": primary,
            "chunk_recall@5": chunk_recall_5,
            "chunk_recall@20": chunk_recall_20,
            "chunk_hits_at_5": chunk_hits_at_5,
            "chunk_hits_at_20": chunk_hits_at_20,
        }

        for k in k_values:
            r = recall_at_k(ranked, list(gold), k)
            p = precision_at_k(ranked, list(gold), k)
            n = ndcg_at_k(ranked, list(gold), k)
            row[f"recall@{k}"] = r
            row[f"precision@{k}"] = p
            row[f"ndcg@{k}"] = n

        m = mrr(ranked, list(gold))
        row["mrr"] = m
        per_case.append(row)
        logger.info(
            "retrieval %s recall@5=%.2f mrr=%.2f chunk_recall@5=%.2f "
            "chunk_recall@20=%.2f primary=%s hits=%d",
            case["id"],
            row.get("recall@5", 0.0),
            m,
            chunk_recall_5,
            chunk_recall_20,
            primary,
            len(hits),
        )

    chunk_gold_rows = [
        c
        for c in per_case
        if str(c.get("category") or "").strip().lower() in CHUNK_GOLD_CATEGORIES
    ]
    excluded_rows = [
        c
        for c in per_case
        if str(c.get("category") or "").strip().lower() in NON_RETRIEVAL_CATEGORIES
    ]
    # Canonical headline averages = chunk-gold categories only (never blend
    # numeric_safety / guardrail / injection / hallucination / out_of_scope).
    averages = _metric_averages(chunk_gold_rows, k_values)
    averages["label"] = "Retrieval quality (chunk-gold categories only)"
    averages["categories"] = sorted(CHUNK_GOLD_CATEGORIES)
    # Legacy all-cases blend kept for diffs only — do not use as the headline.
    averages_all_cases_legacy = _metric_averages(per_case, k_values)
    averages_all_cases_legacy["label"] = (
        "LEGACY blended (all categories) — do not use for production reporting"
    )
    averages_excluded = _metric_averages(excluded_rows, k_values)
    averages_excluded["label"] = (
        "Excluded from retrieval quality (non-retrieval categories)"
    )
    averages_excluded["categories"] = sorted(NON_RETRIEVAL_CATEGORIES)

    enum_rows = [c for c in per_case if c.get("enumerative")]
    by_category: dict[str, Any] = {}
    if enum_rows:
        by_category["enumerative"] = {
            "n_cases": len(enum_rows),
            "primary_metric": "chunk_recall@20",
            "chunk_recall@5": round(
                mean([c.get("chunk_recall@5", 0.0) for c in enum_rows]), 4
            ),
            "chunk_recall@20": round(
                mean([c.get("chunk_recall@20", 0.0) for c in enum_rows]), 4
            ),
            "recall@5": round(mean([c.get("recall@5", 0.0) for c in enum_rows]), 4),
            "mrr": round(mean([c.get("mrr", 0.0) for c in enum_rows]), 4),
            "note": (
                "Enumerative success = fraction of expected_chunk_ids appearing "
                "in the top-20 retrieved chunks (enum breadth), not MRR."
            ),
        }

    return {
        "mode": "retrieval-only",
        "n_cases": len(cases),
        "profile": "pipeline" if use_full_pipeline else "hybrid",
        "k_values": k_values,
        "averages": averages,
        "averages_chunk_gold": averages,
        "averages_excluded_non_retrieval": averages_excluded,
        "averages_all_cases_legacy": averages_all_cases_legacy,
        "by_category": by_category,
        "cases": per_case,
    }
