"""Level 3 per-stage retrieval evaluation — TRD_LEVEL3.md §27-31/§33-39,
Instructions §27-31.

Compares BM25-only, Dense-only, BM25+Dense (RRF, no reranker), and the full
pipeline (RRF + reranker) against the same golden query set
(evals/golden_retrieval_set.yaml), reporting Recall@5, Recall@10, and MRR
for each — plus NDCG@10 comparing RRF-only ordering against the reranked
ordering.

Distinct from evals/retrieval_eval.py (which only reports the full
pipeline's numbers): this is the leg-by-leg comparison TRD_LEVEL3.md §30
requires — "Do not claim that hybrid retrieval improves performance until
measured." Whatever this script prints is what was actually measured; nothing
here is tuned to make hybrid look better.

Usage:
    uv run python evals/level3_hybrid_eval.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import yaml
from sqlalchemy.orm import Session

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.domain.db import get_engine  # noqa: E402
from packages.retrieval.bm25 import build_bm25_index  # noqa: E402
from packages.retrieval.rerank import LexicalAuthorityReranker  # noqa: E402
from packages.retrieval.search import (  # noqa: E402
    SourceFilter,
    full_text_search,
    reciprocal_rank_fusion,
    retrieve,
    vector_search,
)

GOLDEN_SET_PATH = ROOT / "evals" / "golden_retrieval_set.yaml"
K = 10


def _reciprocal_rank(document_keys: list[str], expected: str) -> float:
    for rank, key in enumerate(document_keys, start=1):
        if key == expected:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(document_keys: list[str], expected: str, k: int) -> float:
    """Binary relevance (1 if this result's document is the expected one,
    else 0) — matches what Recall/MRR above already assume about the golden
    set's ground truth (one expected document per query). Only the *first*
    occurrence of the expected document counts: `retrieve()` can return
    several chunks from the same document (capped, not deduped away
    entirely), and NDCG assumes each ranked item is a distinct relevant/
    non-relevant judgment — counting every chunk would let one relevant
    document contribute more than 1.0 of "ideal" relevance and push NDCG
    above 1.0, which is exactly the bug this comment is here to prevent
    reintroducing."""
    first_hit_rank = next((rank for rank, key in enumerate(document_keys[:k], start=1) if key == expected), None)
    dcg = 1.0 / math.log2(first_hit_rank + 1) if first_hit_rank is not None else 0.0
    idcg = 1.0 / math.log2(2)  # the ideal ranking puts the one relevant doc at rank 1
    return dcg / idcg


def _metrics(rankings: list[tuple[list[str], str]]) -> dict[str, float]:
    n = len(rankings)
    recall_5 = sum(1 for keys, exp in rankings if exp in keys[:5]) / n
    recall_10 = sum(1 for keys, exp in rankings if exp in keys[:10]) / n
    mrr = sum(_reciprocal_rank(keys, exp) for keys, exp in rankings) / n
    return {"recall@5": recall_5, "recall@10": recall_10, "mrr": mrr}


def main() -> None:
    with GOLDEN_SET_PATH.open("r", encoding="utf-8") as f:
        golden = yaml.safe_load(f)["cases"]

    with Session(get_engine()) as session:
        bm25_index = build_bm25_index(session)  # built once, reused across every query in this run
        filters = SourceFilter()

        bm25_only: list[tuple[list[str], str]] = []
        dense_only: list[tuple[list[str], str]] = []
        rrf_only: list[tuple[list[str], str]] = []
        full_pipeline: list[tuple[list[str], str]] = []

        for case in golden:
            query, expected = case["query"], case["expected_document_key"]

            bm25_rows = full_text_search(session, query, filters, limit=K, bm25_index=bm25_index)
            bm25_only.append(([r[2].document_key for r in bm25_rows], expected))

            vector_rows = vector_search(session, query, filters, limit=K)
            dense_only.append(([r[2].document_key for r in vector_rows], expected))

            # RRF-only: fuse the same two candidate sets, skip the reranker
            # (a null reranker that returns the fused score unchanged).
            bm25_ids = [r[0].id for r in bm25_rows]
            vector_ids = [r[0].id for r in vector_rows]
            fused = reciprocal_rank_fusion([bm25_ids, vector_ids])
            row_by_id = {r[0].id: r for r in (*bm25_rows, *vector_rows)}
            ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:K]
            rrf_only.append(([row_by_id[cid][2].document_key for cid in ranked_ids], expected))

            full_results = retrieve(session, query, limit=K, bm25_index=bm25_index)
            full_pipeline.append(([r.document_key for r in full_results], expected))

        print(f"Golden set: {len(golden)} queries\n")
        print(f"{'leg':<28} {'recall@5':>10} {'recall@10':>10} {'mrr':>8}")
        print("-" * 60)
        for name, rankings in [
            ("BM25 only", bm25_only),
            ("Dense only", dense_only),
            ("BM25 + Dense (RRF)", rrf_only),
            ("Full pipeline (+ reranker)", full_pipeline),
        ]:
            m = _metrics(rankings)
            print(f"{name:<28} {m['recall@5']:>10.2f} {m['recall@10']:>10.2f} {m['mrr']:>8.3f}")

        ndcg_rrf = sum(_ndcg_at_k(keys, exp, K) for keys, exp in rrf_only) / len(rrf_only)
        ndcg_full = sum(_ndcg_at_k(keys, exp, K) for keys, exp in full_pipeline) / len(full_pipeline)
        print()
        print(f"NDCG@{K} RRF-only:            {ndcg_rrf:.3f}")
        print(f"NDCG@{K} full pipeline+rerank: {ndcg_full:.3f}")

        reranker = LexicalAuthorityReranker()
        print(
            f"\nReranker: {reranker.model_name} {reranker.model_version} (docs/ADR/0012 — real cross-encoder deferred)"
        )
        print(
            "\nNote: retrieval uses the interim HashingEmbeddingProvider (docs/ADR/0007) for the "
            "dense leg — word-overlap only, no semantics. Any BM25 advantage over Dense above "
            "reflects that honestly, not a BM25 implementation bug."
        )


if __name__ == "__main__":
    main()
