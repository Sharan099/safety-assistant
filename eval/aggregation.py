"""Per-category aggregation and threshold gating (shared leaf module).

Used independently by ``eval.run_full``, ``eval.smoke_subset`` (via re-exports /
tests), and ``eval.render_dashboard``. This module must not import those three.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = ROOT / "thresholds.yaml"

_RAGAS_METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    thr_path = path or DEFAULT_THRESHOLDS
    raw = yaml.safe_load(thr_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"thresholds.yaml must be a mapping: {thr_path}")
    return raw


def _is_finite_number(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def mean_metric(vals: Sequence[float]) -> float | None:
    """Arithmetic mean of finite values only (NaN/Inf excluded)."""
    finite = [float(v) for v in vals if _is_finite_number(v)]
    if not finite:
        return None
    return round(sum(finite) / len(finite), 4)


def average_ragas_metric(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Per-metric category average that skips NaN and reports coverage.

    Returns a dict with ``average``, ``n_scored``, ``n_nan``, ``n_missing``,
    ``n_cases``, and a ``display`` string like::

        "0.4909 (31/43 cases scored, 12 NaN)"
    """
    n_cases = len(rows)
    finite: list[float] = []
    n_nan = 0
    n_missing = 0
    for row in rows:
        ragas = row.get("ragas")
        if not isinstance(ragas, dict) or key not in ragas or ragas.get(key) is None:
            n_missing += 1
            continue
        try:
            val = float(ragas[key])
        except (TypeError, ValueError):
            n_missing += 1
            continue
        if math.isnan(val) or math.isinf(val):
            n_nan += 1
            continue
        finite.append(val)

    n_scored = len(finite)
    average = round(sum(finite) / n_scored, 4) if n_scored else None
    if average is None:
        display = f"n/a (0/{n_cases} cases scored, {n_nan} NaN)"
    else:
        display = f"{average} ({n_scored}/{n_cases} cases scored, {n_nan} NaN)"
    return {
        "average": average,
        "n_scored": n_scored,
        "n_nan": n_nan,
        "n_missing": n_missing,
        "n_cases": n_cases,
        "display": display,
    }


def aggregate_per_category(
    case_results: Sequence[dict[str, Any]],
    *,
    false_positive_probe: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute pass_rate and metric averages WITHIN each category only.

    Do NOT compute or store any single blended score across categories.
    Averaging across risk-heterogeneous categories (e.g. mixing factual_lookup
    pass rates with prompt_injection ASR) hides critical failures and is
    explicitly forbidden in this eval design.

    RAGAS category averages skip NaN per metric independently and record how
    many cases contributed vs were excluded as NaN (see ``ragas_coverage``).
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_results:
        cat = str(row.get("category") or "unknown").strip().lower()
        buckets[cat].append(row)

    out: dict[str, dict[str, Any]] = {}
    for cat, rows in sorted(buckets.items()):
        n = len(rows)
        n_pass = sum(1 for r in rows if r.get("pass"))
        agg: dict[str, Any] = {
            "n_cases": n,
            "n_pass": n_pass,
            "pass_rate": round(n_pass / n, 4) if n else None,
        }

        ragas_avgs: dict[str, float | None] = {}
        ragas_coverage: dict[str, dict[str, Any]] = {}
        for key in _RAGAS_METRIC_KEYS:
            cov = average_ragas_metric(rows, key)
            ragas_avgs[key] = cov["average"]
            ragas_coverage[key] = cov
        if any(v is not None for v in ragas_avgs.values()) or any(
            (c.get("n_nan") or 0) > 0 or (c.get("n_scored") or 0) > 0
            for c in ragas_coverage.values()
        ):
            agg["ragas_averages"] = ragas_avgs
            agg["ragas_coverage"] = ragas_coverage

        if cat in {"guardrail", "prompt_injection"}:
            asr_vals = [bool(r.get("attack_succeeded")) for r in rows]
            agg["attack_success_rate"] = (
                round(sum(1 for x in asr_vals if x) / len(asr_vals), 4) if asr_vals else None
            )
            if cat == "guardrail":
                agg["n_breached"] = sum(1 for r in rows if r.get("breached"))
                if false_positive_probe is not None:
                    agg["false_positive_probe"] = {
                        "n_probes": false_positive_probe.get("n_probes"),
                        "false_positives": false_positive_probe.get("false_positives"),
                        "false_positive_rate": false_positive_probe.get(
                            "false_positive_rate"
                        ),
                    }

        if cat == "hallucination_probe":
            deepeval_vals = [
                float(r["deepeval_hallucination"]["score"])
                for r in rows
                if isinstance(r.get("deepeval_hallucination"), dict)
                and r["deepeval_hallucination"].get("score") is not None
                and _is_finite_number(r["deepeval_hallucination"]["score"])
            ]
            if deepeval_vals:
                agg["avg_deepeval_hallucination_score"] = mean_metric(deepeval_vals)

        out[cat] = agg

    return out


def apply_thresholds(
    per_category: dict[str, dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Compare each category aggregate to thresholds.yaml; set overall_status."""
    cat_cfg = thresholds.get("categories") or {}
    category_results: dict[str, dict[str, Any]] = {}
    critical_failures: list[str] = []
    noncritical_failures: list[str] = []

    for cat, agg in sorted(per_category.items()):
        cfg = cat_cfg.get(cat) or {}
        severity = str(cfg.get("severity") or "medium").strip().lower()
        failures: list[str] = []

        min_pr = cfg.get("min_pass_rate")
        pr = agg.get("pass_rate")
        if min_pr is not None and pr is not None and float(pr) < float(min_pr):
            failures.append(f"pass_rate {pr} < min_pass_rate {min_pr}")

        max_asr = cfg.get("max_attack_success_rate")
        asr = agg.get("attack_success_rate")
        if max_asr is not None and asr is not None and float(asr) > float(max_asr):
            failures.append(f"attack_success_rate {asr} > max_attack_success_rate {max_asr}")

        max_fpr = cfg.get("max_false_positive_rate")
        fp = (agg.get("false_positive_probe") or {}).get("false_positive_rate")
        if max_fpr is not None and fp is not None and float(fp) > float(max_fpr):
            failures.append(
                f"false_positive_rate {fp} > max_false_positive_rate {max_fpr}"
            )

        ragas_avgs = agg.get("ragas_averages") or {}
        for thr_key, metric in (
            ("min_avg_faithfulness", "faithfulness"),
            ("min_avg_answer_relevancy", "answer_relevancy"),
            ("min_avg_context_precision", "context_precision"),
            ("min_avg_context_recall", "context_recall"),
        ):
            floor = cfg.get(thr_key)
            val = ragas_avgs.get(metric)
            if floor is not None and val is not None and float(val) < float(floor):
                failures.append(f"avg_{metric} {val} < {thr_key} {floor}")

        met = len(failures) == 0
        category_results[cat] = {
            "severity": severity,
            "met_threshold": met,
            "failures": failures,
            "aggregate": agg,
            "thresholds_applied": {
                k: cfg[k]
                for k in (
                    "min_pass_rate",
                    "max_attack_success_rate",
                    "max_false_positive_rate",
                    "min_avg_faithfulness",
                    "min_avg_answer_relevancy",
                    "min_avg_context_precision",
                    "min_avg_context_recall",
                )
                if k in cfg
            },
        }
        if not met:
            if severity == "critical":
                critical_failures.append(cat)
            else:
                noncritical_failures.append(cat)

    if critical_failures:
        overall = "NOT PRODUCTION READY"
    elif noncritical_failures:
        overall = "NEEDS IMPROVEMENT"
    else:
        overall = "PRODUCTION READY"

    return {
        "overall_status": overall,
        "critical_failures": critical_failures,
        "noncritical_failures": noncritical_failures,
        "categories": category_results,
    }


def load_partial_results(path: Path) -> list[dict[str, Any]]:
    """Read previously-completed case rows from ``partial_results.jsonl`` (resume / dashboard)."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt partial_results.jsonl line (%d chars)", len(line))
    return rows


def compute_gap_set(
    golden_cases: Sequence[dict[str, Any]],
    result_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the gap set vs an authoritative golden list.

    A case is in the gap set when either:

    - it has **no result row** at all (never attempted / missing from the run), or
    - it has a result with ``pass`` not true (genuine scored failure **or** an
      unrecoverable error row that recorded ``pass=False`` + ``error`` without a
      real answer).

    Returns a report dict with ``gap_ids``, reason breakdowns, per-category
    counts, and the golden case dicts for the gap subset (``gap_cases``).
    """
    by_id_gold: dict[str, dict[str, Any]] = {}
    for case in golden_cases:
        cid = str(case.get("id") or "").strip()
        if cid:
            by_id_gold[cid] = case

    result_map: dict[str, dict[str, Any]] = {}
    for row in result_cases:
        cid = str(row.get("id") or row.get("case_id") or "").strip()
        if not cid:
            continue
        # Last write wins if duplicates appear in partial merges.
        result_map[cid] = row

    missing_ids: list[str] = []
    error_ids: list[str] = []
    failed_ids: list[str] = []
    passed_ids: list[str] = []
    gap_rows: list[dict[str, Any]] = []
    by_category: dict[str, int] = defaultdict(int)
    by_category_reason: dict[str, dict[str, int]] = defaultdict(
        lambda: {"missing": 0, "error": 0, "failed": 0}
    )

    for cid, case in by_id_gold.items():
        cat = str(case.get("category") or "unknown").strip().lower() or "unknown"
        row = result_map.get(cid)
        if row is None:
            reason = "missing"
            missing_ids.append(cid)
            by_category[cat] += 1
            by_category_reason[cat]["missing"] += 1
            gap_rows.append(
                {
                    "id": cid,
                    "category": cat,
                    "severity": case.get("severity"),
                    "reason": reason,
                    "pass": None,
                    "has_result": False,
                    "error": None,
                }
            )
            continue

        has_result = True
        passed = bool(row.get("pass"))
        err = row.get("error")
        err_s = str(err).strip() if err is not None else ""
        if passed:
            passed_ids.append(cid)
            continue

        # Unrecoverable failures often store pass=False + error and no usable answer.
        if err_s:
            reason = "error"
            error_ids.append(cid)
            by_category_reason[cat]["error"] += 1
        else:
            reason = "failed"
            failed_ids.append(cid)
            by_category_reason[cat]["failed"] += 1
        by_category[cat] += 1
        gap_rows.append(
            {
                "id": cid,
                "category": cat,
                "severity": case.get("severity"),
                "reason": reason,
                "pass": False,
                "has_result": has_result,
                "error": err_s or None,
            }
        )

    gap_ids = [r["id"] for r in gap_rows]
    gap_cases = [by_id_gold[cid] for cid in gap_ids if cid in by_id_gold]

    return {
        "n_golden": len(by_id_gold),
        "n_results": len(result_map),
        "n_gap": len(gap_ids),
        "n_passed": len(passed_ids),
        "n_missing": len(missing_ids),
        "n_error": len(error_ids),
        "n_failed": len(failed_ids),
        "gap_ids": gap_ids,
        "missing_ids": missing_ids,
        "error_ids": error_ids,
        "failed_ids": failed_ids,
        "passed_ids": passed_ids,
        "gap_rows": gap_rows,
        "gap_cases": gap_cases,
        "by_category": dict(sorted(by_category.items())),
        "by_category_reason": {
            cat: dict(reasons) for cat, reasons in sorted(by_category_reason.items())
        },
        "result_map": result_map,
    }


class GapMergeError(ValueError):
    """Merged case list does not match the golden set frame (count / dupes / gaps)."""


def merge_fill_gaps_results(
    golden_cases: Sequence[dict[str, Any]],
    *,
    parent_cases: Sequence[dict[str, Any]],
    gap_ids: Sequence[str],
    new_gap_results: Sequence[dict[str, Any]] | dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge a completed fill-gaps run onto the full golden-set frame.

    - Frame = every case in ``golden_cases`` (authoritative order / membership).
    - Cases **not** in ``gap_ids`` keep the parent row unchanged.
    - Cases **in** ``gap_ids`` take the new fill-gaps result.

    Raises ``GapMergeError`` if the merge would leave missing ids, duplicates, or a
    count that does not equal ``len(golden_cases)``.
    """
    gold_ids: list[str] = []
    gold_seen: set[str] = set()
    for case in golden_cases:
        cid = str(case.get("id") or "").strip()
        if not cid:
            continue
        if cid in gold_seen:
            raise GapMergeError(f"Duplicate id in golden_set: {cid}")
        gold_seen.add(cid)
        gold_ids.append(cid)

    gap_id_set = {str(x).strip() for x in gap_ids if str(x).strip()}
    if not gap_id_set.issubset(gold_seen):
        unknown = sorted(gap_id_set - gold_seen)
        raise GapMergeError(
            f"gap_ids not present in golden_set ({len(unknown)}): {', '.join(unknown[:20])}"
        )

    parent_map: dict[str, dict[str, Any]] = {}
    for row in parent_cases:
        cid = str(row.get("id") or row.get("case_id") or "").strip()
        if cid:
            parent_map[cid] = dict(row)

    if isinstance(new_gap_results, dict):
        new_map = {str(k): dict(v) for k, v in new_gap_results.items()}
    else:
        new_map = {}
        for row in new_gap_results:
            cid = str(row.get("id") or row.get("case_id") or "").strip()
            if cid:
                new_map[cid] = dict(row)

    missing_new = sorted(gap_id_set - set(new_map))
    if missing_new:
        raise GapMergeError(
            "fill-gaps merge incomplete — new results missing for gap id(s): "
            + ", ".join(missing_new[:20])
            + (" ..." if len(missing_new) > 20 else "")
        )

    merged: list[dict[str, Any]] = []
    kept_ids: list[str] = []
    rescored_ids: list[str] = []
    missing_from_parent: list[str] = []

    for cid in gold_ids:
        if cid in gap_id_set:
            merged.append(new_map[cid])
            rescored_ids.append(cid)
            continue
        parent_row = parent_map.get(cid)
        if parent_row is None:
            missing_from_parent.append(cid)
            continue
        merged.append(parent_row)
        kept_ids.append(cid)

    if missing_from_parent:
        raise GapMergeError(
            "Non-gap golden cases missing from parent results (cannot keep unchanged): "
            + ", ".join(missing_from_parent[:20])
            + (" ..." if len(missing_from_parent) > 20 else "")
        )

    assert_merged_matches_golden(merged, gold_ids)
    return {
        "cases": merged,
        "kept_ids": kept_ids,
        "rescored_ids": rescored_ids,
        "n_golden": len(gold_ids),
        "n_merged": len(merged),
        "gap_ids": sorted(gap_id_set),
    }


def assert_merged_matches_golden(
    merged_cases: Sequence[dict[str, Any]],
    golden_ids: Sequence[str],
) -> None:
    """Fail loudly unless merged cases == golden ids (count, set, no duplicates)."""
    expected = [str(x).strip() for x in golden_ids if str(x).strip()]
    got = [str(r.get("id") or r.get("case_id") or "").strip() for r in merged_cases]
    got = [x for x in got if x]

    if len(got) != len(expected):
        raise GapMergeError(
            f"Merged case count {len(got)} != golden_set count {len(expected)}"
        )
    if len(set(got)) != len(got):
        dupes = sorted({x for x in got if got.count(x) > 1})
        raise GapMergeError(
            f"Merged results contain duplicate case ids ({len(dupes)}): "
            + ", ".join(dupes[:20])
        )
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing={missing[:20]}")
        if extra:
            parts.append(f"extra={extra[:20]}")
        raise GapMergeError(
            "Merged results do not match golden_set membership: " + "; ".join(parts)
        )
    # Preserve golden order exactly.
    if got != expected:
        raise GapMergeError(
            "Merged results are not in golden_set order "
            f"(first mismatch at index "
            f"{next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)})"
        )
