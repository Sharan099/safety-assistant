#!/usr/bin/env python3
"""Run regulation test-case scorecard (retrieval-only or full with cached answers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.eval_harness.scoring import aggregate_pass_rates, score_item  # noqa: E402
from tests.score_golden_set import _get_retrieval_pipeline, _retrieve_documents  # noqa: E402


def main() -> int:
    path = ROOT / "tests" / "test_cases_regulation.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    retriever, reranker = _get_retrieval_pipeline()
    lookup = retriever._chunk_by_id

    rows = []
    for case in cases:
        docs = _retrieve_documents(case["question"], retriever, reranker)
        row = score_item(case, "", docs, lookup, require_answer=False)
        retr_ok = all(
            row.get(k) in ("PASS", "skip")
            for k in ("recall", "must_not", "retrieval_contains")
        )
        row["retrieval_pass"] = retr_ok
        rows.append(row)

    agg = aggregate_pass_rates(
        [{**r, "pass": r["retrieval_pass"]} for r in rows]
    )
    print(f"\nRegulation retrieval regression: {agg['overall']['pass']}/{agg['overall']['total']}")
    for r in rows:
        status = "PASS" if r["retrieval_pass"] else "FAIL"
        print(
            f"  {r['id']}: {status} | recall={r['recall']} "
            f"retrieval_contains={r.get('retrieval_contains', 'n/a')}"
        )
    return 0 if agg["overall"]["rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
