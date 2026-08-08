"""LEGACY fixture helper — not the CI gate.

Fails if recall@5 / faithfulness drop below thresholds on an old scorecard JSON.
Authoritative CI is ``python -m eval.run_full`` writing ``results.json`` and
enforcing ``eval/thresholds.yaml`` (see ``.github/workflows/quality-gate.yml``).
``main.yml`` still smoke-imports this module against a synthetic fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_RECALL = 0.55
DEFAULT_FAITHFULNESS = 0.70


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Fail CI when eval metrics regress.")
    p.add_argument("--scorecard", type=Path, default=Path("eval/results/latest.json"))
    p.add_argument("--min-recall-at-5", type=float, default=float(os.getenv("GATE_MIN_RECALL_AT_5", DEFAULT_RECALL)))
    p.add_argument(
        "--min-faithfulness",
        type=float,
        default=float(os.getenv("GATE_MIN_FAITHFULNESS", DEFAULT_FAITHFULNESS)),
    )
    p.add_argument(
        "--require-faithfulness",
        action="store_true",
        help="Fail if faithfulness metric is missing (use on full-eval jobs).",
    )
    args = p.parse_args(argv)

    if not args.scorecard.is_file():
        print(f"FAIL: scorecard not found: {args.scorecard}", file=sys.stderr)
        return 2

    data = json.loads(args.scorecard.read_text(encoding="utf-8-sig"))
    sc = data.get("scorecard") or {}
    retrieval = data.get("retrieval") or {}
    retrieval_avg = retrieval.get("averages") or {}

    recall = (
        sc.get("recall@5")
        or retrieval_avg.get("recall@5")
        or data.get("recall@5")
        or data.get("recall_at_5")
    )
    faithfulness = (
        sc.get("deepeval_faithfulness")
        or sc.get("ragas_faithfulness")
        or sc.get("faithfulness")
        or (data.get("deepeval") or {}).get("averages", {}).get("faithfulness")
        or (data.get("ragas") or {}).get("averages", {}).get("faithfulness")
        or data.get("faithfulness")
    )

    failures: list[str] = []
    if recall is None:
        failures.append("recall@5 missing from scorecard")
    elif float(recall) < args.min_recall_at_5:
        failures.append(f"recall@5={recall} < gate {args.min_recall_at_5}")

    if faithfulness is None:
        if args.require_faithfulness:
            failures.append("faithfulness missing (required)")
        else:
            print("WARN: faithfulness not in scorecard — skipping faithfulness gate")
    elif float(faithfulness) < args.min_faithfulness:
        failures.append(f"faithfulness={faithfulness} < gate {args.min_faithfulness}")

    print(
        json.dumps(
            {
                "scorecard": str(args.scorecard),
                "recall@5": recall,
                "faithfulness": faithfulness,
                "gates": {
                    "min_recall_at_5": args.min_recall_at_5,
                    "min_faithfulness": args.min_faithfulness,
                },
                "ok": not failures,
                "failures": failures,
            },
            indent=2,
        )
    )
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("PASS: quality gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
