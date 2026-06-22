#!/usr/bin/env python3
"""
Stage A — Deterministic golden-set scorer (ZERO LLM tokens).

Usage:
  EVAL_SKIP_LLM=true python tests/score_golden_set.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EVALUATION_CURRENT
from tests.eval_harness.cache import CACHE_FILE, load_cache
from tests.eval_harness.golden import load_items
from tests.eval_harness.scoring import aggregate_pass_rates, score_item


def _get_retrieval_pipeline():
    from backend.app.core.services import get_retriever, get_reranker
    return get_retriever(), get_reranker()


def _retrieve_documents(question: str, retriever, reranker) -> list[dict]:
    result = retriever.retrieve(question)
    docs = result["documents"]
    if reranker:
        rr = reranker.rerank(question, docs)
        docs = rr["documents"]
    return docs


def run_deterministic_scorecard(
    *,
    golden_path: Path | None = None,
    cache_path: Path | None = None,
    require_answer: bool | None = None,
) -> dict:
    items = load_items(golden_path)
    cache = load_cache(cache_path)
    skip_llm = os.getenv("EVAL_SKIP_LLM", "").lower() in ("1", "true", "yes")
    if require_answer is None:
        require_answer = False  # produce scorecard even without cached answers

    retriever, reranker = _get_retrieval_pipeline()
    chunk_lookup = retriever._chunk_by_id

    rows = []
    for item in items:
        cached = (cache.get("items") or {}).get(item["id"], {})
        answer = cached.get("answer", "")
        if cached.get("documents"):
            documents = cached["documents"]
        else:
            documents = _retrieve_documents(item["question"], retriever, reranker)

        row = score_item(
            item,
            answer,
            documents,
            chunk_lookup,
            require_answer=require_answer,
        )
        rows.append(row)

    aggregate = aggregate_pass_rates(rows)
    scorecard = {
        "stage": "deterministic",
        "eval_skip_llm": skip_llm,
        "items": rows,
        "aggregate": aggregate,
    }

    out_path = EVALUATION_CURRENT / "deterministic_scorecard.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
    return scorecard


def _print_table(scorecard: dict) -> None:
    print("\nDeterministic scorecard (Stage A — zero tokens)")
    print("id | type | recall | must_not | behavior | contains | forbidden | PASS/FAIL")
    print("-" * 90)
    for r in scorecard["items"]:
        status = "PASS" if r.get("pass") else "FAIL"
        print(
            f"{r['id']} | {r.get('query_type', '')} | {r.get('recall')} | "
            f"{r.get('must_not')} | {r.get('behavior')} | {r.get('contains')} | "
            f"{r.get('forbidden')} | {status}"
        )

    agg = scorecard["aggregate"]
    print(f"\nOverall: {agg['overall']['pass']}/{agg['overall']['total']} "
          f"({agg['overall']['rate']:.1%})")
    for qt, stats in agg.get("by_query_type", {}).items():
        print(f"  {qt}: {stats['pass']}/{stats['total']} ({stats['rate']:.1%})")
    print(f"\nWrote {EVALUATION_CURRENT / 'deterministic_scorecard.json'}")


def main() -> int:
    scorecard = run_deterministic_scorecard()
    _print_table(scorecard)
    return 0 if scorecard["aggregate"]["overall"]["rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
