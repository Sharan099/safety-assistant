"""CLI: run the citation-strict regulation agent.

Usage::

    python -m agent.run "Compare R94 vs R95 chest deflection limits"
    python -m agent.run "Gap analysis: our test setup vs R95 requirements" --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from agent.loop import run_agent
from generation.llm_client import LLMClient
from ingestion.hf_auth import ensure_hf_auth


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agent.run")
    p.add_argument("task", help="Multi-step agent task")
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_human(result) -> None:
    print("=" * 64)
    print(f"MODE={result.mode}  coverage={result.overall_citation_coverage:.2f}  "
          f"ungrounded={result.ungrounded_claim_count}")
    print("=" * 64)
    print("PLAN")
    for i, step in enumerate(result.plan, 1):
        print(f"  {i}. {step.tool} {step.args} — {step.rationale}")
    print()
    print("STEPS")
    for st in result.steps:
        print(
            f"  [{st.tool}] cov={st.citation_coverage:.2f} "
            f"cites={len(st.citations)} err={st.error}"
        )
    print()
    if result.table_markdown:
        print("TABLE")
        print(result.table_markdown)
        print()
    if result.report_markdown and result.report_markdown != result.answer:
        print("REPORT")
        print(result.report_markdown)
        print()
    print("ANSWER")
    print(result.answer)
    print()
    print(f"trace_id={result.trace_id} provider={result.provider}")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ensure_hf_auth()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result = run_agent(args.task, llm=LLMClient())
    if args.json:
        print(result.to_json())
    else:
        _print_human(result)
    # Fail only when we asserted facts without citations (not when a side is unindexed).
    if result.ungrounded_claim_count > 0 and result.overall_citation_coverage < 0.9:
        sys.exit(2)


if __name__ == "__main__":
    main()
