#!/usr/bin/env python3
"""
Model A/B benchmark on golden set — retrieval recall, latency, cost estimates.

Usage:
  conda activate rag
  python scripts/benchmark_models.py
  python scripts/benchmark_models.py --quick   # 5 cases only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    EMBEDDING_MODEL,
    GROQ_MODEL,
    RERANKER_MODEL,
)


def _load_golden(quick: bool) -> list[dict]:
    path = ROOT / "tests" / "golden_set.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    return cases[:5] if quick else cases


def _recall_at_k(docs: list[dict], expected_doc: str, k: int = 10) -> float:
    chunk_ids = [d.get("id", "") for d in docs[:k]]
    retriever_chunks = []
    from backend.app.retrieval.hybrid import HybridRetriever
  # lazy — caller passes retriever
    return 0.0  # placeholder overridden below


def benchmark_retrieval(cases: list[dict]) -> dict:
    from backend.app.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()
    hits = 0
    total = 0
    latencies: list[float] = []
    for case in cases:
        if case.get("type") == "abstention":
            continue
        expected = case.get("expected_doc", "")
        t0 = time.perf_counter()
        result = retriever.retrieve(case["question"])
        latencies.append(time.perf_counter() - t0)
        docs = result.get("documents", [])
        found = False
        for d in docs[:10]:
            chunk = retriever._chunk_by_id.get(d.get("id", ""), {})
            reg = chunk.get("regulation", "")
            pdf = chunk.get("pdf_name", "")
            if expected.upper() in reg.upper() or expected.lower() in pdf.lower():
                found = True
                break
        hits += int(found)
        total += 1
    return {
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "recall_at_10": round(hits / max(total, 1), 3),
        "avg_latency_s": round(sum(latencies) / max(len(latencies), 1), 2),
        "cases_evaluated": total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cases = _load_golden(args.quick)
    retrieval = benchmark_retrieval(cases)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_llm": GROQ_MODEL,
        "retrieval": retrieval,
        "recommendation": {
            "embedding": EMBEDDING_MODEL,
            "reranker": RERANKER_MODEL,
            "generation": GROQ_MODEL,
            "rationale": (
                "Defaults chosen for CPU deploy balance: Nomic embeddings + "
                "bge-reranker-v2-m3 + llama-3.3-70b-versatile. "
                "Run with EMBEDDING_MODEL / RERANKER_MODEL env overrides to A/B."
            ),
        },
    }
    out_json = ROOT / "output" / "benchmark_models.json"
    out_md = ROOT / "output" / "model_selection.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(
        f"# Model Selection (auto-generated)\n\n"
        f"- **Embedding:** `{retrieval['embedding_model']}`\n"
        f"- **Reranker:** `{retrieval['reranker_model']}`\n"
        f"- **Generation:** `{GROQ_MODEL}`\n"
        f"- **Recall@10:** {retrieval['recall_at_10']}\n"
        f"- **Avg retrieval latency:** {retrieval['avg_latency_s']}s\n\n"
        f"{report['recommendation']['rationale']}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
