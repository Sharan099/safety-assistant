"""Smoke subset: run the FULL pipeline on 5 representative golden-set cases.

Selects one case each from:
  - ``factual_lookup``
  - ``compliance_check``
  - ``numeric_safety``
  - ``enumerative`` or ``multi_hop`` (whichever appears first in the golden set)
  - a critical category: ``prompt_injection`` or ``hallucination_probe``

...and runs them through the exact same dispatch used by the full run
(``eval.case_scoring.score_one_case``) — retrieval, generation, RAGAS, custom hard
gates, and DeepEval/DeepTeam security scoring, depending on category. Nothing
here is mocked or shortcut; the only difference from a full run is the number
of cases (5 instead of the canonical **30** in ``eval/golden_set.jsonl``).

Purpose: catch pipeline wiring bugs, scoring logic errors, timeout
misconfigurations, and provider issues for the price of ~5 cases worth of LLM
calls, before spending quota on the remaining ~25 questions.

Usage::

    python -m eval.smoke_subset
    python -m eval.smoke_subset --skip-ragas

Also runs automatically as the default step of ``python -m eval.run_full``
(see ``--smoke-only`` / ``--smoke-first`` / ``--skip-smoke`` there).

Scoring dispatch lives in ``eval.case_scoring`` (not ``eval.run_full``) so this
module and ``run_full`` do not import each other. Per-category gates live in
``eval.aggregation``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

# DeepTeam (Python 3.13) still imports removed stdlib nntplib — shim before scorers.
from eval.nntplib_shim import ensure_nntplib_shim

ensure_nntplib_shim()

from eval.aggregation import aggregate_per_category
from eval.gold import DEFAULT_GOLDEN, load_golden_set
from eval.scoring.security_scorer import SECURITY_CATEGORIES
from generation.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Each tuple is a "slot" to fill; the first category in the tuple that has an
# unused case in the golden set wins. Order matches the user-facing spec.
SMOKE_CATEGORY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("factual_lookup",),
    ("compliance_check",),
    ("numeric_safety",),
    ("enumerative", "multi_hop"),
    ("prompt_injection", "hallucination_probe"),
)


def select_smoke_cases(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick one representative case per ``SMOKE_CATEGORY_GROUPS`` slot.

    Raises if the golden set is missing a case for any required slot — a smoke
    subset that silently drops a category defeats its own purpose.
    """
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        cat = str(c.get("category") or "").strip().lower()
        by_cat.setdefault(cat, []).append(c)

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    missing: list[str] = []
    for group in SMOKE_CATEGORY_GROUPS:
        picked = None
        for cat in group:
            candidates = [c for c in by_cat.get(cat, []) if str(c.get("id")) not in seen_ids]
            if candidates:
                picked = candidates[0]
                break
        if picked is None:
            missing.append(" or ".join(group))
            continue
        selected.append(picked)
        seen_ids.add(str(picked.get("id")))

    if missing:
        raise RuntimeError(
            "Smoke subset: golden set has no case for required categor"
            + ("y" if len(missing) == 1 else "ies")
            + ": "
            + ", ".join(missing)
        )
    return selected


def _print_case_result(row: dict[str, Any]) -> None:
    status = "PASS" if row.get("pass") else "FAIL"
    cost = row.get("_smoke_cost") or {}
    print(
        f"  -> {status}  time={row.get('_smoke_timing_seconds')}s  "
        f"calls={cost.get('calls', 0)}  cost=${float(cost.get('cost_usd', 0.0)):.4f}  "
        f"tokens(in/out)={cost.get('input_tokens', 0)}/{cost.get('output_tokens', 0)}"
    )
    if row.get("error"):
        print(f"     ERROR: {row['error']}")
    print(json.dumps(row, indent=2, ensure_ascii=False, default=str))
    print()


def run_smoke_subset(
    *,
    gold_path: Path | None = None,
    llm: LLMClient | None = None,
    skip_ragas: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the 5-case smoke subset through the real pipeline; return a summary dict."""
    from eval.case_scoring import (
        _category_cost_entry,
        _usage_snapshot,
        score_one_case,
        summarize_usage_since,
    )
    from eval.case_timeout import case_timeout_enabled, score_case_with_timeout

    all_cases = load_golden_set(gold_path or DEFAULT_GOLDEN)
    cases = select_smoke_cases(all_cases)

    client = llm or LLMClient()
    log_path = Path(client.log_path)

    security_judge = None
    guard_in = guard_out = None
    if any(str(c.get("category") or "").strip().lower() in SECURITY_CATEGORIES for c in cases):
        from eval.scoring.security_scorer import _build_topic_guards, _portkey_security_judge

        security_judge = _portkey_security_judge(client)
        guard_in, guard_out = _build_topic_guards(security_judge)

    run_start_size = _usage_snapshot(log_path)
    wall_t0 = time.perf_counter()
    results: list[dict[str, Any]] = []

    for i, case in enumerate(cases, 1):
        cat = str(case.get("category") or "").strip().lower()
        cid = case.get("id")
        if verbose:
            print(f"[{i}/{len(cases)}] smoke: {cid} ({cat}) ...")
        case_log_start = _usage_snapshot(log_path)
        t0 = time.perf_counter()
        try:
            if case_timeout_enabled():
                row = score_case_with_timeout(
                    case,
                    skip_ragas=skip_ragas,
                    with_security=security_judge is not None,
                    llm_provider=str(getattr(client, "provider", None) or "") or None,
                )
            else:
                row = score_one_case(
                    case,
                    llm=client,
                    skip_ragas=skip_ragas,
                    security_judge=security_judge,
                    guard_input=guard_in,
                    guard_output=guard_out,
                )
        except Exception as exc:  # noqa: BLE001 — a smoke failure must not crash the check
            logger.exception("smoke case %s failed", cid)
            row = {
                "id": cid,
                "category": cat,
                "severity": case.get("severity"),
                "pass": False,
                "error": str(exc),
            }
        elapsed = time.perf_counter() - t0
        case_log_end = _usage_snapshot(log_path)
        usage = summarize_usage_since(log_path, start_size=case_log_start, end_size=case_log_end)
        cost_entry = _category_cost_entry(usage)
        row["_smoke_timing_seconds"] = round(elapsed, 3)
        row["_smoke_cost"] = cost_entry
        results.append(row)
        if verbose:
            _print_case_result(row)

    wall_clock_seconds = time.perf_counter() - wall_t0
    overall_usage = summarize_usage_since(log_path, start_size=run_start_size)
    n_pass = sum(1 for r in results if r.get("pass"))
    per_category = aggregate_per_category(results)

    return {
        "n_cases": len(results),
        "n_pass": n_pass,
        "all_passed": n_pass == len(results),
        "cases": results,
        "per_category": per_category,
        "wall_clock_seconds": round(wall_clock_seconds, 3),
        "total_cost_usd": overall_usage["total_cost_usd"],
        "total_tokens": overall_usage["total_tokens"],
        "total_input_tokens": overall_usage["total_input_tokens"],
        "total_output_tokens": overall_usage["total_output_tokens"],
    }


def print_smoke_summary(summary: dict[str, Any]) -> None:
    dur = float(summary.get("wall_clock_seconds") or 0.0)
    dur_s = f"{dur / 60:.1f}m" if dur >= 60 else f"{dur:.1f}s"
    print("=" * 72)
    print(f"SMOKE SUBSET: {summary['n_pass']}/{summary['n_cases']} passed  (duration={dur_s})")
    print(
        f"  cost=${float(summary.get('total_cost_usd') or 0.0):.4f}  "
        f"tokens={int(summary.get('total_tokens') or 0):,} "
        f"(in={int(summary.get('total_input_tokens') or 0):,} / "
        f"out={int(summary.get('total_output_tokens') or 0):,})"
    )
    for row in summary["cases"]:
        status = "PASS" if row.get("pass") else "FAIL"
        note = f" — {row['error']}" if row.get("error") else ""
        print(
            f"  [{status}] {row.get('id')} ({row.get('category')})  "
            f"time={row.get('_smoke_timing_seconds')}s{note}"
        )
    print("=" * 72)


def confirm_smoke_or_abort(summary: dict[str, Any], *, assume_yes: bool = False) -> bool:
    """Return True if it's OK to proceed into the full batched run."""
    if summary.get("all_passed"):
        return True

    print(
        "\nWARNING: smoke subset had failure(s) — the full run is likely to reproduce "
        "the same wiring/scoring/provider issue at scale, burning quota on ~145 more cases.",
        file=sys.stderr,
    )
    for row in summary["cases"]:
        if not row.get("pass"):
            detail = f": {row['error']}" if row.get("error") else ""
            print(f"  - {row.get('id')} ({row.get('category')}) FAILED{detail}", file=sys.stderr)

    if assume_yes:
        print("Proceeding anyway (--yes / EVAL_ASSUME_YES set).", file=sys.stderr)
        return True
    if not sys.stdin.isatty():
        print(
            "stdin is not a TTY (non-interactive run) — pass --yes to proceed despite "
            "the smoke failure(s) above, or fix the issue and re-run.",
            file=sys.stderr,
        )
        return False
    try:
        reply = input("Continue into the full run anyway? [y/N] ").strip().lower()
    except EOFError:
        reply = ""
    return reply in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full RAG + scoring pipeline on 5 representative golden-set cases"
    )
    p.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Path to golden_set.jsonl (default: eval/golden_set.jsonl, 30-case canonical set)",
    )
    p.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS LLM-judge metrics (substring + hard gates still run)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    summary = run_smoke_subset(gold_path=args.gold, skip_ragas=args.skip_ragas)
    print_smoke_summary(summary)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
