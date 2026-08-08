"""Retrieval-only eval (zero LLM cost) — CLI over ``eval.retrieval_eval``.

Preferred path for retrieval scorecards after ``eval.run`` was removed.
Full generation + per-category gates: ``python -m eval.run_full``.

Usage::

    python -m eval.run_retrieval_only
    python -m eval.run_retrieval_only --tag baseline --profile hybrid
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from eval.retrieval_eval import run_retrieval_eval
from eval.scorecard import build_scorecard, print_scorecard, save_scorecard

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Zero-LLM retrieval-only evaluation against golden_set.jsonl"
    )
    p.add_argument("--tag", default="retrieval-only", help="Scorecard tag / label")
    p.add_argument(
        "--profile",
        default="hybrid",
        choices=["dense", "hybrid", "rerank", "pipeline", "full"],
        help="Retrieval profile (default: hybrid, no rewrite LLM)",
    )
    p.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Override path to golden_set.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional scorecard output path (default: eval/results via save_scorecard)",
    )
    p.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Limit eval to one or more case ids (repeatable)",
    )
    args = p.parse_args(argv)

    retrieval = run_retrieval_eval(
        gold_path=args.gold,
        profile=args.profile,
        case_ids=args.case_id,
    )
    scorecard = build_scorecard(
        retrieval=retrieval,
        ragas=None,
        deepeval=None,
        tag=args.tag,
    )
    print_scorecard(scorecard)
    path = save_scorecard(scorecard, out=args.out)
    logger.info("Wrote retrieval-only scorecard to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
