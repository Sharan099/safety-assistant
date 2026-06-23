#!/usr/bin/env python3
"""
Full evaluation harness: answer cache + Stage A + budgeted RAGAS + combined report.

Usage:
  # Full run (generate + score + RAGAS):
  python tests/run_ragas_eval.py --ragas-subset 5 --judge-delay 3

  # Re-score from cache, no judge tokens:
  python tests/run_ragas_eval.py --from-cache --skip-judge

  # Stage A only via score_golden_set.py:
  EVAL_SKIP_LLM=true python tests/score_golden_set.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.eval_harness.cache import build_or_load_answers, load_cache
from tests.eval_harness.generation import generate_answer_for_item
from tests.eval_harness.golden import (
    ABSTENTION_IDS,
    DEFAULT_RAGAS_SUBSET_IDS,
    load_items,
)
from tests.eval_harness.ragas_budget import run_ragas_budgeted
from tests.eval_harness.gateway_stats import aggregate_gateway_tier_stats
from tests.eval_harness.report import build_eval_report, plot_eval_charts, write_eval_report
from tests.score_golden_set import run_deterministic_scorecard


def _parse_subset_ids(raw: str | None, n: int) -> list[str]:
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return DEFAULT_RAGAS_SUBSET_IDS[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Groq-budgeted RAGAS evaluation harness")
    parser.add_argument("--ragas-subset", type=int, default=int(os.getenv("RAGAS_SUBSET", "5")))
    parser.add_argument("--judge-delay", type=float, default=float(os.getenv("JUDGE_DELAY_SECONDS", "3")))
    parser.add_argument("--token-budget", type=int, default=int(os.getenv("RAGAS_TOKEN_BUDGET", "8000")))
    parser.add_argument("--from-cache", action="store_true", help="Do not generate missing answers")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-judged RAGAS metrics")
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Generate answers and run judge even when EVAL_SKIP_LLM=true in env",
    )
    parser.add_argument("--subset-ids", type=str, default=os.getenv("RAGAS_SUBSET_IDS", ""))
    parser.add_argument("--golden", type=Path, default=None)
    args = parser.parse_args()

    items = load_items(args.golden)
    subset_ids = _parse_subset_ids(args.subset_ids or None, args.ragas_subset)

    skip_llm = os.getenv("EVAL_SKIP_LLM", "").lower() in ("1", "true", "yes")
    if args.force_llm:
        skip_llm = False
    gen_delay = args.judge_delay if not args.from_cache else 0.0

    if not skip_llm and not args.from_cache:
        workflow = None
        retriever = None
        reranker = None

        def _gen(item):
            nonlocal workflow, retriever, reranker
            return generate_answer_for_item(item, workflow, retriever, reranker)

        # Lazy-init inside first call
        from backend.app.core.services import get_retriever, get_reranker, get_workflow
        retriever = get_retriever()
        reranker = get_reranker()
        workflow = get_workflow()

        cache = build_or_load_answers(
            items,
            generate_fn=_gen,
            from_cache_only=False,
            skip_llm=False,
            delay_seconds=gen_delay,
        )
    else:
        cache = build_or_load_answers(
            items,
            generate_fn=lambda _: {},
            from_cache_only=args.from_cache or skip_llm,
            skip_llm=skip_llm,
        )

    # Stage A — require cached answers when generation ran
    has_answers = any(
        bool((cache.get("items") or {}).get(i["id"], {}).get("answer"))
        for i in items
    )
    scorecard = run_deterministic_scorecard(require_answer=has_answers)

    gateway_stats = aggregate_gateway_tier_stats(cache.get("items") or {})

    # Stage B — local answer_relevancy always; judge metrics unless --skip-judge / EVAL_SKIP_LLM
    ragas_result = run_ragas_budgeted(
        items,
        cache,
        subset_ids=subset_ids,
        abstention_ids=ABSTENTION_IDS,
        token_budget=args.token_budget,
        judge_delay=args.judge_delay,
        skip_judge=args.skip_judge or (skip_llm and not args.force_llm),
    )

    report = build_eval_report(
        deterministic=scorecard,
        ragas=ragas_result,
        meta={
            "from_cache": args.from_cache,
            "skip_judge": args.skip_judge,
            "force_llm": args.force_llm,
            "subset_ids": subset_ids,
            "gateway_tier_stats": gateway_stats,
        },
    )
    json_path, md_path = write_eval_report(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    png_paths = plot_eval_charts(report)
    for p in png_paths:
        print(f"Wrote {p}")

    if ragas_result:
        ts = ragas_result.get("token_summary", {})
        print(
            f"Groq judge: {ts.get('groq_calls', 0)} calls, "
            f"{ts.get('total_tokens', 0)} tokens "
            f"(budget {ts.get('budget', 0)})"
        )

    return 0 if scorecard["aggregate"]["overall"]["rate"] == 1.0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise
