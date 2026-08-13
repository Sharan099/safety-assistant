"""P0 embedding benchmark — PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md §5/§21.

Benchmarks real candidate embedding models against the interim hashing
placeholder, using the dense leg alone (packages/retrieval/search.py's
vector_search()) over the golden retrieval set. The doc's suggested
candidates (BGE-M3, an E5-family model, a Qwen-embedding model) are all
>2 GB — substituted here with small, hardware-appropriate real semantic
models per the *same* doc's own hardware rules (§20: "no giant local model
merely to claim SOTA," "CPU-first"): `BAAI/bge-small-en-v1.5` (the BGE
family, small variant) and `sentence-transformers/all-MiniLM-L6-v2` (a
second, architecturally distinct small real candidate). Both are real
semantic ONNX models via `fastembed` (docs/ADR/0014) — not the hashing
placeholder, not a heavier substitute that risks the disk exhaustion
docs/ADR/0011 already measured for `torch`-based alternatives.

Usage:
    uv run python evals/embedding_benchmark.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import tracemalloc
from typing import Any

import yaml
from sqlalchemy.orm import Session

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.domain.db import get_engine  # noqa: E402
from packages.retrieval.embeddings import EmbeddingProvider, FastEmbedProvider, HashingEmbeddingProvider  # noqa: E402
from packages.retrieval.index import index_chunks  # noqa: E402
from packages.retrieval.search import SourceFilter, vector_search  # noqa: E402

GOLDEN_SET_PATH = ROOT / "evals" / "golden_retrieval_set.yaml"
RESULTS_PATH = ROOT / "evals" / "results" / "embedding_benchmark.json"
K = 10


def _reciprocal_rank(document_keys: list[str], expected: str) -> float:
    for rank, key in enumerate(document_keys, start=1):
        if key == expected:
            return 1.0 / rank
    return 0.0


def _evaluate(session: Session, provider: EmbeddingProvider, golden: list[dict[str, Any]]) -> dict[str, float]:
    filters = SourceFilter()
    rankings: list[tuple[list[str], str]] = []
    latencies: list[float] = []
    for case in golden:
        t0 = time.perf_counter()
        rows = vector_search(session, case["query"], filters, limit=K, provider=provider)
        latencies.append(time.perf_counter() - t0)
        keys = [r[2].document_key for r in rows]
        rankings.append((keys, case["expected_document_key"]))

    n = len(rankings)
    recall_5 = sum(1 for keys, exp in rankings if exp in keys[:5]) / n
    recall_10 = sum(1 for keys, exp in rankings if exp in keys[:10]) / n
    mrr = sum(_reciprocal_rank(keys, exp) for keys, exp in rankings) / n
    return {
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr": mrr,
        "avg_query_latency_ms": (sum(latencies) / n) * 1000,
    }


def main() -> None:
    with GOLDEN_SET_PATH.open("r", encoding="utf-8") as f:
        golden = yaml.safe_load(f)["cases"]

    candidates: list[tuple[str, EmbeddingProvider]] = [
        ("hashing-bow (baseline, mock tier)", HashingEmbeddingProvider()),
        ("BAAI/bge-small-en-v1.5 (local tier candidate)", FastEmbedProvider("BAAI/bge-small-en-v1.5")),
        (
            "sentence-transformers/all-MiniLM-L6-v2 (local tier candidate)",
            FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2"),
        ),
    ]

    results: list[dict[str, Any]] = []
    with Session(get_engine()) as session:
        for label, provider in candidates:
            tracemalloc.start()
            t0 = time.perf_counter()
            # Idempotent (packages/retrieval/index.py): only embeds chunks
            # not yet embedded under this exact (model_name, model_version)
            # — the baseline is typically already fully embedded from a
            # prior run, so its indexing_time/newly_embedded here will be
            # ~0, which is expected and correctly reported, not an error.
            embedded_count = index_chunks(session, provider)
            index_time = time.perf_counter() - t0
            _, peak_ram = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            metrics = _evaluate(session, provider, golden)
            record = {
                "label": label,
                "model_name": provider.model_name,
                "model_version": provider.model_version,
                "dimensions": provider.dimensions,
                "newly_embedded_chunks": embedded_count,
                "indexing_time_s": round(index_time, 3),
                "peak_ram_bytes_during_indexing": peak_ram,
                **metrics,
            }
            results.append(record)
            print(
                f"{label:60s} recall@5={metrics['recall@5']:.2f} recall@10={metrics['recall@10']:.2f} "
                f"mrr={metrics['mrr']:.3f} latency={metrics['avg_query_latency_ms']:.1f}ms"
            )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "hardware": "Intel i5-8250U, 8 GB RAM, GeForce 940MX 2 GB (PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md header)",
                "golden_set_size": len(golden),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {RESULTS_PATH}")

    real_candidates = [r for r in results if r["model_name"] != "hashing-bow"]
    baseline = next(r for r in results if r["model_name"] == "hashing-bow")
    if real_candidates:
        winner = max(real_candidates, key=lambda r: (r["mrr"], r["recall@10"]))
        verdict = "beats" if winner["mrr"] > baseline["mrr"] else "does NOT beat"
        print(f"\nBest real candidate: {winner['label']} — {verdict} the hashing baseline.")
        print(f"MRR: {winner['mrr']:.3f} vs {baseline['mrr']:.3f}")


if __name__ == "__main__":
    main()
