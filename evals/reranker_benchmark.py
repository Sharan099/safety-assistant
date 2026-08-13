"""P1 reranker benchmark — PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md §6/§21.

Compares LexicalAuthorityReranker (mock tier) against real cross-encoder
candidates on NDCG@10, MRR, Recall@10, and latency — reranking the same
real RRF-fused candidate set (BM25 + real semantic dense, docs/ADR/0014)
for every golden-set query. "Reject a candidate that does not improve
ranking quality on the domain golden set" (§6) — this prints the actual
measurement, the production default only changes if a candidate earns it.

Usage:
    uv run python evals/reranker_benchmark.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from typing import Any

import yaml
from sqlalchemy.orm import Session

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.domain.db import get_engine  # noqa: E402
from packages.retrieval.bm25 import build_bm25_index  # noqa: E402
from packages.retrieval.rerank import (  # noqa: E402
    CrossEncoderReranker,
    LexicalAuthorityReranker,
    RerankCandidate,
    Reranker,
    rerank,
)
from packages.retrieval.search import (  # noqa: E402
    SourceFilter,
    full_text_search,
    reciprocal_rank_fusion,
    vector_search,
)

GOLDEN_SET_PATH = ROOT / "evals" / "golden_retrieval_set.yaml"
RESULTS_PATH = ROOT / "evals" / "results" / "reranker_benchmark.json"
K = 10


def _ndcg_at_k(document_keys: list[str], expected: str, k: int) -> float:
    first_hit_rank = next((rank for rank, key in enumerate(document_keys[:k], start=1) if key == expected), None)
    dcg = 1.0 / math.log2(first_hit_rank + 1) if first_hit_rank is not None else 0.0
    return dcg / (1.0 / math.log2(2))


def _reciprocal_rank(document_keys: list[str], expected: str) -> float:
    for r, key in enumerate(document_keys, start=1):
        if key == expected:
            return 1.0 / r
    return 0.0


def _fused_candidates(session: Session, query: str, bm25_index: Any) -> list[tuple[RerankCandidate, str]]:
    """Real RRF-fused candidates for one query — document_key kept
    alongside each RerankCandidate for scoring against the golden answer."""
    filters = SourceFilter()
    bm25_rows = full_text_search(session, query, filters, limit=K, bm25_index=bm25_index)
    vector_rows = vector_search(session, query, filters, limit=K)
    bm25_ids = [r[0].id for r in bm25_rows]
    vector_ids = [r[0].id for r in vector_rows]
    fused = reciprocal_rank_fusion([bm25_ids, vector_ids])
    row_by_id = {r[0].id: r for r in (*bm25_rows, *vector_rows)}

    candidates = []
    for chunk_id, score in fused.items():
        chunk, _rev, document, ks, _section = row_by_id[chunk_id]
        candidates.append(
            (
                RerankCandidate(
                    id=chunk.id, content=chunk.content, authority_level=ks.authority_level, fused_score=score
                ),
                document.document_key,
            )
        )
    return candidates


def _evaluate(session: Session, reranker: Reranker, golden: list[dict[str, Any]], bm25_index: Any) -> dict[str, float]:
    rankings: list[tuple[list[str], str]] = []
    latencies: list[float] = []

    for case in golden:
        pairs = _fused_candidates(session, case["query"], bm25_index)
        candidates = [c for c, _key in pairs]
        key_by_id = {c.id: key for c, key in pairs}

        t0 = time.perf_counter()
        observation = rerank(reranker, case["query"], candidates)
        latencies.append(time.perf_counter() - t0)

        document_keys = [key_by_id[r.id] for r in observation.results]
        rankings.append((document_keys, case["expected_document_key"]))

    n = len(rankings)
    recall_10 = sum(1 for keys, exp in rankings if exp in keys[:10]) / n
    mrr = sum(_reciprocal_rank(keys, exp) for keys, exp in rankings) / n
    ndcg_10 = sum(_ndcg_at_k(keys, exp, K) for keys, exp in rankings) / n
    return {
        "recall@10": recall_10,
        "mrr": mrr,
        "ndcg@10": ndcg_10,
        "avg_latency_ms": (sum(latencies) / n) * 1000,
    }


def main() -> None:
    with GOLDEN_SET_PATH.open("r", encoding="utf-8") as f:
        golden = yaml.safe_load(f)["cases"]

    candidates: list[Reranker] = [
        LexicalAuthorityReranker(),
        CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2"),
        CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-12-v2"),
    ]

    results: list[dict[str, Any]] = []
    with Session(get_engine()) as session:
        bm25_index = build_bm25_index(session)
        for reranker in candidates:
            metrics = _evaluate(session, reranker, golden, bm25_index)
            record = {"model_name": reranker.model_name, "model_version": reranker.model_version, **metrics}
            results.append(record)
            print(
                f"{reranker.model_name:35s} ndcg@10={metrics['ndcg@10']:.3f} mrr={metrics['mrr']:.3f} "
                f"recall@10={metrics['recall@10']:.2f} latency={metrics['avg_latency_ms']:.1f}ms"
            )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"golden_set_size": len(golden), "results": results}, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {RESULTS_PATH}")

    baseline = results[0]
    best_real = max(results[1:], key=lambda r: (r["ndcg@10"], r["mrr"]))
    verdict = "beats" if best_real["ndcg@10"] > baseline["ndcg@10"] else "does NOT beat"
    print(f"\nBest cross-encoder: {best_real['model_name']} — {verdict} the lexical/authority baseline on NDCG@10.")
    print(f"NDCG@10: {best_real['ndcg@10']:.3f} vs {baseline['ndcg@10']:.3f}")


if __name__ == "__main__":
    main()
