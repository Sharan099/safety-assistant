"""Aggregate an existing partial_results.jsonl into results.json (no LLM)."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.aggregation import DEFAULT_THRESHOLDS, load_partial_results, load_thresholds
from eval.run_full import (
    RESULTS_DIR,
    _accumulate_case_cost,
    _empty_category_cost_entry,
    _finalize_merged_results,
    _new_cat_usage_bucket,
    _overall_usage_from_per_category,
    build_cost_summary,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    rid = str(args.run_id).strip()
    out_dir = RESULTS_DIR / rid
    partial = out_dir / "partial_results.jsonl"
    if not partial.is_file():
        raise SystemExit(f"missing {partial}")

    rows = load_partial_results(partial)
    print(f"loaded {len(rows)} cases from {partial}")
    thresholds = load_thresholds(DEFAULT_THRESHOLDS)
    per_cat: dict[str, dict] = defaultdict(_new_cat_usage_bucket)
    for r in rows:
        cat = str(r.get("category") or "unknown").strip().lower() or "unknown"
        entry = r.get("_case_cost") or _empty_category_cost_entry()
        _accumulate_case_cost(per_cat[cat], entry)
    overall = _overall_usage_from_per_category(per_cat)
    cost = build_cost_summary(
        overall=overall,
        per_category={k: dict(v) for k, v in sorted(per_cat.items())},
        wall_clock_seconds=0.0,
    )
    cost["note"] = (
        "finalize-from-partial: zero LLM; costs from partial case rows only; "
        "FP probe not completed"
    )
    results, code = _finalize_merged_results(
        rid=rid,
        out_dir=out_dir,
        case_results=rows,
        parent_run_id=rid,
        mode="finalize_from_partial",
        gold_path=None,
        thresholds_path=None,
        thresholds=thresholds,
        fp_report=None,
        cost_summary=cost,
        estimate={"total": 0, "note": "aggregation only"},
        rescored_ids=[],
        kept_ids=[str(r.get("id")) for r in rows],
    )
    print(f"wrote {out_dir / 'results.json'}")
    print(f"overall_status={results.get('overall_status')} exit={code}")
    for cat, agg in sorted((results.get("per_category") or {}).items()):
        print(
            f"  {cat}: n={agg.get('n_cases')} pass_rate={agg.get('pass_rate')} "
            f"asr={agg.get('attack_success_rate')}"
        )
    gate = results.get("gate") or {}
    print(f"critical_failures={gate.get('critical_failures')}")
    print(f"noncritical_failures={gate.get('noncritical_failures')}")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
