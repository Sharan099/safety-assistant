"""Collect RAG answers for generation metrics + Groq-call estimates."""

from __future__ import annotations

import logging
from typing import Any

from generation.answer import answer_question
from generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


def estimate_groq_calls(
    n_cases: int,
    *,
    provider: str,
    rewrite_enabled: bool = True,
    use_ragas: bool = True,
    use_deepeval: bool = True,
) -> dict[str, int | str]:
    """Estimate billable Groq chat calls for a full generation eval."""
    if provider != "groq":
        return {
            "rewrite": 0,
            "answer": 0,
            "ragas_judge": 0,
            "deepeval_judge": 0,
            "total": 0,
            "note": "LLM_PROVIDER is not groq — generation judges will be skipped or mock",
        }
    rewrite = n_cases if rewrite_enabled else 0
    answer = n_cases
    # RAGAS runs ~several judge prompts per metric per case; rough lower bound:
    ragas = (4 * n_cases) if use_ragas else 0
    # DeepEval: 4 metrics × ~1-3 calls; use 4*n as lower bound.
    deepeval = (4 * n_cases) if use_deepeval else 0
    return {
        "rewrite": rewrite,
        "answer": answer,
        "ragas_judge": ragas,
        "deepeval_judge": deepeval,
        "total": rewrite + answer + ragas + deepeval,
        "note": "Judge counts are lower bounds; frameworks may issue multiple calls per metric",
    }


def collect_generation_records(
    gold_cases: list[dict[str, Any]],
    *,
    llm: LLMClient | None = None,
    skip_abstention: bool = True,
) -> list[dict[str, Any]]:
    """Run the RAG answer pipeline for each gold case; return RAGAS/DeepEval rows."""
    client = llm or LLMClient()
    records: list[dict[str, Any]] = []
    for case in gold_cases:
        if skip_abstention and case.get("abstention"):
            continue
        question = case["question"]
        ans = answer_question(
            question,
            regulation_id=case.get("regulation_id"),
            llm=client,
        )
        contexts = [s.text for s in ans.sources if (s.text or "").strip()]
        if not contexts:
            contexts = ["(no retrieved context)"]
        records.append(
            {
                "id": case["id"],
                "question": question,
                "answer": ans.answer,
                "ground_truth": case.get("answer") or case.get("ground_truth") or "",
                "contexts": contexts,
                "source_citations": [s.citation for s in ans.sources],
                "sources": [s.model_dump() for s in ans.sources],
                "provider": ans.provider,
                "model": ans.model,
            }
        )
        logger.info("collected generation %s sources=%d", case["id"], len(ans.sources))
    return records
