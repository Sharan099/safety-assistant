"""Determinism check — same question N times must retrieve the same chunk set.

Category: ``determinism``. Default probe is the checklist question that previously
drifted across runs when LLM rewrite ran twice / at non-zero temperature.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from generation.llm_client import LLMClient
from retrieval.retrieve import retrieve
from retrieval.rewrite import rewrite_query

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DEFAULT_PROBE = (
    "Generate a checklist for preparing a vehicle for UN R94 testing"
)


def _chunk_fingerprint(chunks: list[Any]) -> tuple[str, ...]:
    return tuple(c.chunk_id for c in chunks if getattr(c, "chunk_id", None))


def run_determinism_check(
    *,
    question: str = DEFAULT_PROBE,
    n: int = 5,
    regulation_id: str | None = "UN-ECE-R94",
    llm: LLMClient | None = None,
    use_llm_rewrite: bool = False,
) -> dict[str, Any]:
    """Run retrieve N times; require identical ordered chunk-id lists.

    Uses the production path: ``rewrite_query`` once → ``retrieve(..., rewrite=False,
    rewrite_result=...)`` so we do not double-call the rewrite LLM.
    """
    n = max(2, int(n))
    client = llm or LLMClient(provider=os.getenv("LLM_PROVIDER") or "mock")
    # Determinism eval always forces heuristic subquery split unless explicitly enabled.
    prev = os.environ.get("RETRIEVAL_REWRITE_LLM")
    if not use_llm_rewrite:
        os.environ["RETRIEVAL_REWRITE_LLM"] = "0"
    # Prefer exact ANN for the probe (small corpus).
    prev_exact = os.environ.get("QDRANT_EXACT_SEARCH")
    os.environ.setdefault("QDRANT_EXACT_SEARCH", "1")

    runs: list[dict[str, Any]] = []
    try:
        for i in range(n):
            rewritten = rewrite_query(question, llm=client, history=None)
            chunks = retrieve(
                rewritten.condensed or question,
                regulation_id=regulation_id,
                llm=client,
                history=None,
                rewrite=False,
                rewrite_result=rewritten,
            )
            fp = _chunk_fingerprint(chunks)
            runs.append(
                {
                    "run": i + 1,
                    "condensed": rewritten.condensed,
                    "expanded": rewritten.expanded,
                    "subqueries": list(rewritten.subqueries),
                    "chunk_ids": list(fp),
                    "scores": [
                        {"chunk_id": c.chunk_id, "score": round(float(c.score or 0.0), 6)}
                        for c in chunks
                        if c.chunk_id
                    ],
                }
            )
            logger.info(
                "determinism run=%d subqueries=%s chunks=%s",
                i + 1,
                rewritten.subqueries,
                list(fp),
            )
    finally:
        if prev is None:
            os.environ.pop("RETRIEVAL_REWRITE_LLM", None)
        else:
            os.environ["RETRIEVAL_REWRITE_LLM"] = prev
        if prev_exact is None:
            os.environ.pop("QDRANT_EXACT_SEARCH", None)
        else:
            os.environ["QDRANT_EXACT_SEARCH"] = prev_exact

    fingerprints = [tuple(r["chunk_ids"]) for r in runs]
    subquery_fps = [tuple(r["subqueries"]) for r in runs]
    chunks_stable = len(set(fingerprints)) == 1
    queries_stable = len(set(subquery_fps)) == 1
    passed = chunks_stable and queries_stable and bool(fingerprints[0])

    report = {
        "category": "determinism",
        "question": question,
        "n": n,
        "passed": passed,
        "chunks_identical": chunks_stable,
        "subqueries_identical": queries_stable,
        "canonical_chunk_ids": list(fingerprints[0]) if fingerprints else [],
        "runs": runs,
    }
    if not passed:
        # Diff first disagreement for the scorecard.
        for i in range(1, len(fingerprints)):
            if fingerprints[i] != fingerprints[0] or subquery_fps[i] != subquery_fps[0]:
                report["first_diff_run"] = i + 1
                report["diff"] = {
                    "run1_subqueries": list(subquery_fps[0]),
                    "runN_subqueries": list(subquery_fps[i]),
                    "run1_chunks": list(fingerprints[0]),
                    "runN_chunks": list(fingerprints[i]),
                }
                break
    return report


def assert_determinism(report: dict[str, Any]) -> None:
    if report.get("passed"):
        return
    raise AssertionError(
        "Determinism check failed: same question retrieved different chunks/subqueries. "
        + json.dumps(
            {k: report.get(k) for k in ("first_diff_run", "diff", "runs")},
            ensure_ascii=False,
            indent=2,
        )[:4000]
    )
