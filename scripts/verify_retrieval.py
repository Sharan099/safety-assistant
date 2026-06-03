#!/usr/bin/env python3
"""
verify_retrieval.py — Smoke test for the hybrid + rerank chain.

Checks:
  1. BM25 sparse retrieval
  2. Semantic dense retrieval (BAAI/bge-base-en-v1.5)
  3. Reciprocal Rank Fusion (RRF) of both
  4. Cross-encoder reranking (BAAI/bge-reranker-base)

Run:  python scripts/verify_retrieval.py
"""

from __future__ import annotations

import os

# Must be set before torch/sentence-transformers import (Windows OpenMP fix).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

QUERIES = [
    "What test load is applied to safety-belt anchorages for strength?",
    "Dynamic test requirements for seat belt assemblies",
]


def banner(text: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {text}")
    print("=" * 64)


def main() -> None:
    from backend.app.retrieval.hybrid import HybridRetriever
    from backend.app.retrieval.reranker import CrossEncoderReranker

    banner("Loading retriever")
    retriever = HybridRetriever()

    reranker = CrossEncoderReranker()

    for query in QUERIES:
        banner(f"Query: {query}")

        # Stage isolation
        allowed = retriever._filter_chunk_ids(retriever._detect_regs(query))
        bm25 = retriever._bm25_search(query, allowed)
        semantic = retriever._semantic_search(query, allowed)
        fused = retriever._rrf_fusion(semantic, bm25, k=60)

        print(f"  BM25 hits      : {len(bm25)}")
        print(f"  Semantic hits  : {len(semantic)}  "
              f"(disabled={retriever._semantic_disabled})")
        print(f"  RRF fused      : {len(fused)}")

        reranked = reranker.rerank(query, fused)
        print(f"  Reranker used  : {reranked['reranker_used']} "
              f"({reranked['latency_ms']} ms)")

        print("\n  Top results after rerank:")
        for i, d in enumerate(reranked["documents"][:5], 1):
            score = d.get("rerank_score", d.get("score", 0))
            head = (d.get("heading_path") or d.get("title") or "")[:70]
            src = d.get("source", "?")
            print(f"   {i}. [{src:8s} score={score:.3f}] {head}")
            snippet = d.get("text", "").replace("\n", " ")[:120]
            print(f"      {snippet}")

    banner("Verification complete")
    health = {
        "chunks": len(retriever.chunks),
        "vectors": len(retriever._matrix_chunk_ids),
        "bm25_ready": retriever._bm25 is not None,
        "semantic_disabled": retriever._semantic_disabled,
        "reranker_enabled": reranker._enabled,
        "reranker_model": os.getenv("RERANKER_MODEL", "default"),
    }
    for k, v in health.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
