"""Per-query retrieval/synthesis timing (mirrors ingest_log timed_stage pattern, in-memory)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from loguru import logger

STEP_KEYS = (
    "guardrails_ms",
    "dense_retrieval_ms",
    "sparse_retrieval_ms",
    "rrf_fusion_ms",
    "rerank_ms",
    "annex_promotion_ms",
    "parent_expansion_ms",
    "cross_reference_expansion_ms",
    "llm_generation_ms",
)


class QueryTiming:
    """Accumulates per-step milliseconds for a single search() call."""

    def __init__(self) -> None:
        self.steps: dict[str, float] = {k: 0.0 for k in STEP_KEYS}
        self.rerank_bypassed = False

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        if name not in self.steps:
            raise KeyError(f"unknown timing step: {name}")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.steps[name] += (time.perf_counter() - t0) * 1000

    def add(self, name: str, ms: float) -> None:
        if name not in self.steps:
            raise KeyError(f"unknown timing step: {name}")
        self.steps[name] += ms

    def parts_sum_ms(self) -> float:
        return sum(self.steps.values())

    def finalize(self, total_ms: float) -> dict:
        parts_sum = self.parts_sum_ms()
        overhead_ms = max(0.0, total_ms - parts_sum)
        slowest_key, slowest_val = max(self.steps.items(), key=lambda kv: kv[1])
        return {
            **{k: round(v, 2) for k, v in self.steps.items()},
            "rerank_bypassed": self.rerank_bypassed,
            "overhead_ms": round(overhead_ms, 2),
            "parts_sum_ms": round(parts_sum, 2),
            "total_ms": round(total_ms, 2),
            "slowest_step": slowest_key.replace("_ms", ""),
            "slowest_ms": round(slowest_val, 2),
        }


def log_timing_summary(query: str, timing: dict) -> None:
    """One-line per-request timing for real-time ops visibility."""
    q = query[:80].replace("\n", " ")
    logger.info(
        "Query timing q='{}' total={}ms slowest={} ({}ms) | guardrails={} dense={} "
        "sparse={} rrf={} rerank={} annex={} parent={} llm={} overhead={} "
        "rerank_bypassed={}",
        q,
        timing["total_ms"],
        timing["slowest_step"],
        timing["slowest_ms"],
        timing["guardrails_ms"],
        timing["dense_retrieval_ms"],
        timing["sparse_retrieval_ms"],
        timing["rrf_fusion_ms"],
        timing["rerank_ms"],
        timing["annex_promotion_ms"],
        timing["parent_expansion_ms"],
        timing["llm_generation_ms"],
        timing["overhead_ms"],
        timing.get("rerank_bypassed"),
    )
