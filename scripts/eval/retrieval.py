"""Run the retrieval evaluation per leg and write evals/results/retrieval_<dataset>_<ts>.json.

uv run python scripts/eval/retrieval.py                       # all legs, regulatory_v1
uv run python scripts/eval/retrieval.py --legs full sparse    # subset
uv run python scripts/eval/retrieval.py --dataset evals/datasets/regulatory_v1.yaml --slices
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from sqlalchemy.orm import Session

from safety_assistant.evaluation import LEGS, format_summary, load_dataset, run_evaluation, write_report
from safety_assistant.persistence import get_engine


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="evals/datasets/regulatory_v1.yaml")
    ap.add_argument("--legs", nargs="*", choices=list(LEGS), default=None)
    ap.add_argument("--out", default="evals/results")
    ap.add_argument("--slices", action="store_true", help="print per-query_type table for the last leg")
    ap.add_argument("--source", choices=["human", "llm_generated", "llm_generated_reviewed"], default=None)
    ap.add_argument("--types", nargs="*", default=None, help="restrict to these query_type values")
    args = ap.parse_args(argv)

    dataset = load_dataset(pathlib.Path(args.dataset), source=args.source, query_types=args.types or None)
    with Session(get_engine()) as session:
        report = run_evaluation(session, dataset, legs=args.legs)
    path = write_report(report, pathlib.Path(args.out))
    print(format_summary(report))
    if args.slices:
        leg = report.legs[-1]
        print(f"\nper query_type ({leg.leg}):")
        for qt, a in leg.by_slice.items():
            mrr = "n/a" if a["mrr"] is None else f"{a['mrr']:.3f}"
            r10 = "n/a" if a["recall@10"] is None else f"{a['recall@10']:.3f}"
            print(f"  {qt:30s} n={int(a['n'] or 0):2d}  R@10={r10:>5s}  MRR={mrr:>5s}")
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
