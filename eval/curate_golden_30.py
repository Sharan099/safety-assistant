"""Curate a 30-case honest subset from eval/golden_set.jsonl.

Builds ``eval/golden_set_30.jsonl`` with exact per-category quotas, using the
merged fill-gaps run for current pass/fail and the parent full run to detect
cases that were recently fixed (fail → pass).

Selection goals (honesty over cosmetics):
  - ≥1 currently PASSING case per category
  - ≥1 CURRENTLY FAILING **or** recently fixed case per category
    (when such a case exists in that category)
  - Prefer cases diagnosed in project history (num_003/005, xrg_008, dsn_001, …)

Usage::

    python -m eval.curate_golden_30
    python -m eval.curate_golden_30 --dry-run   # table only, do not write JSONL

Review the printed table before treating the 30-set as final.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "eval" / "golden_set.jsonl"
DEFAULT_OUT = ROOT / "eval" / "golden_set_30.jsonl"
DEFAULT_MERGED = (
    ROOT / "eval" / "results" / "20260806T111508Z_from_20260805T152635Z" / "results.json"
)
DEFAULT_PARENT = ROOT / "eval" / "results" / "20260805T152635Z" / "results.json"

CATEGORY_TARGETS: dict[str, int] = {
    "compliance_check": 3,
    "numeric_safety": 3,
    "prompt_injection": 3,
    "cross_regulation": 3,
    "guardrail": 3,
    "factual_lookup": 3,
    "design_implication": 3,
    "hallucination_probe": 3,
    "multi_hop": 2,
    "enumerative": 2,
    "out_of_scope": 2,
}
assert sum(CATEGORY_TARGETS.values()) == 30

# Interview-defensible / diagnosed cases first within each category.
PRIORITY_IDS: dict[str, list[str]] = {
    "numeric_safety": ["num_003", "num_005", "num_004"],
    "cross_regulation": ["xrg_008", "xrg_001", "xrg_002"],
    "design_implication": ["dsn_001", "dsn_005", "dsn_002"],
    "prompt_injection": ["pin_004", "pin_001", "pin_002"],
    "factual_lookup": ["fac_001", "fac_002", "fac_003"],
    "hallucination_probe": ["hal_001", "hal_002", "hal_011"],
    "multi_hop": ["mhp_001", "mhp_003"],
    "enumerative": ["enm_001", "enm_002"],
    "guardrail": ["grd_001", "grd_006", "grd_002"],
    # No fail / recently-fixed in parent→merged for these categories.
    "compliance_check": ["cmp_004", "cmp_001", "cmp_003"],
    "out_of_scope": ["oos_001", "oos_002"],
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _load_results_map(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") or data.get("results") or []
    return {str(c["id"]): c for c in cases if c.get("id")}


def _status_label(
    *,
    currently_pass: bool | None,
    recently_fixed: bool,
) -> str:
    if currently_pass is None:
        return "MISSING"
    if currently_pass and recently_fixed:
        return "PASS (recently fixed)"
    if currently_pass:
        return "PASS"
    return "FAIL"


def _classify(
    gold: list[dict[str, Any]],
    merged: dict[str, dict[str, Any]],
    parent: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return case_id -> meta with pass/fail/recently_fixed and gold row."""
    meta: dict[str, dict[str, Any]] = {}
    for g in gold:
        cid = str(g["id"])
        cat = str(g.get("category") or "").strip()
        mr = merged.get(cid)
        pr = parent.get(cid)
        cur_pass = bool(mr.get("pass")) if mr is not None else None
        parent_pass = bool(pr.get("pass")) if pr is not None else None
        recently_fixed = parent_pass is False and cur_pass is True
        meta[cid] = {
            "id": cid,
            "category": cat,
            "gold": g,
            "currently_pass": cur_pass,
            "recently_fixed": recently_fixed,
            "challenging": (cur_pass is False) or recently_fixed,
            "status": _status_label(
                currently_pass=cur_pass, recently_fixed=recently_fixed
            ),
        }
    return meta


def _select_category(
    cat: str,
    n: int,
    by_cat: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Pick n cases for one category; return (selected, warnings)."""
    warnings: list[str] = []
    pool = list(by_cat.get(cat) or [])
    if len(pool) < n:
        raise ValueError(f"{cat}: only {len(pool)} cases in golden set, need {n}")

    by_id = {m["id"]: m for m in pool}
    priority = [cid for cid in PRIORITY_IDS.get(cat, []) if cid in by_id]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(m: dict[str, Any]) -> None:
        if m["id"] in selected_ids:
            return
        if len(selected) >= n:
            return
        selected.append(m)
        selected_ids.add(m["id"])

    for cid in priority:
        add(by_id[cid])

    # Ensure ≥1 currently passing.
    if selected and not any(m["currently_pass"] for m in selected):
        passer = next((m for m in pool if m["currently_pass"] and m["id"] not in selected_ids), None)
        if passer is not None:
            # Replace last non-priority filler if full, else append.
            if len(selected) >= n:
                # Prefer swapping a non-challenging priority if possible; else last.
                swap_i = next(
                    (
                        i
                        for i, m in enumerate(selected)
                        if not m["challenging"] and m["id"] not in priority[:1]
                    ),
                    len(selected) - 1,
                )
                old = selected[swap_i]
                selected_ids.discard(old["id"])
                selected[swap_i] = passer
                selected_ids.add(passer["id"])
            else:
                add(passer)

    # Ensure ≥1 failing or recently fixed when available in the category.
    has_challenging_in_pool = any(m["challenging"] for m in pool)
    if has_challenging_in_pool and not any(m["challenging"] for m in selected):
        chal = next(
            (m for m in pool if m["challenging"] and m["id"] not in selected_ids),
            None,
        )
        if chal is not None:
            if len(selected) >= n:
                # Prefer replacing a non-priority, currently-passing case.
                swap_i = next(
                    (
                        i
                        for i, m in enumerate(selected)
                        if m["currently_pass"]
                        and not m["recently_fixed"]
                        and m["id"] not in PRIORITY_IDS.get(cat, [])[:1]
                    ),
                    len(selected) - 1,
                )
                old = selected[swap_i]
                selected_ids.discard(old["id"])
                selected[swap_i] = chal
                selected_ids.add(chal["id"])
            else:
                add(chal)
    elif not has_challenging_in_pool:
        warnings.append(
            f"{cat}: no FAIL or recently-fixed cases in merged+parent — "
            "mix rule only partially satisfiable (all selected are long-standing PASS)"
        )

    # Fill remaining slots: prefer challenging not yet picked, then passers, then rest.
    remaining = [m for m in pool if m["id"] not in selected_ids]
    remaining.sort(
        key=lambda m: (
            0 if m["challenging"] else 1,
            0 if m["currently_pass"] else 1,
            m["id"],
        )
    )
    for m in remaining:
        if len(selected) >= n:
            break
        add(m)

    if len(selected) != n:
        raise ValueError(f"{cat}: selected {len(selected)}, need {n}")

    # Final mix checks.
    if not any(m["currently_pass"] for m in selected):
        warnings.append(f"{cat}: selected set has no currently PASSING case")
    if has_challenging_in_pool and not any(m["challenging"] for m in selected):
        warnings.append(f"{cat}: selected set has no FAIL / recently-fixed case")

    # Stable order: priority order first, then id.
    prio_rank = {cid: i for i, cid in enumerate(PRIORITY_IDS.get(cat, []))}
    selected.sort(key=lambda m: (prio_rank.get(m["id"], 999), m["id"]))
    return selected, warnings


def curate(
    *,
    golden_path: Path = DEFAULT_GOLDEN,
    merged_path: Path = DEFAULT_MERGED,
    parent_path: Path = DEFAULT_PARENT,
) -> tuple[list[dict[str, Any]], list[str]]:
    gold = _load_jsonl(golden_path)
    merged = _load_results_map(merged_path)
    parent = _load_results_map(parent_path) if parent_path.is_file() else {}
    meta = _classify(gold, merged, parent)

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in meta.values():
        by_cat[m["category"]].append(m)

    selected: list[dict[str, Any]] = []
    warnings: list[str] = []
    for cat, n in CATEGORY_TARGETS.items():
        picks, w = _select_category(cat, n, by_cat)
        selected.extend(picks)
        warnings.extend(w)

    if len(selected) != 30:
        raise RuntimeError(f"Expected 30 cases, got {len(selected)}")
    return selected, warnings


def write_jsonl(selected: list[dict[str, Any]], out_path: Path) -> None:
    lines = [json.dumps(m["gold"], ensure_ascii=False) for m in selected]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_review_table(selected: list[dict[str, Any]], warnings: list[str]) -> None:
    print()
    print("Proposed golden_set_30 (review before treating as final)")
    print(f"Source results: {DEFAULT_MERGED.parent.name}")
    print(f"{'ID':<12} {'Category':<22} {'Status':<24}")
    print("-" * 60)
    for m in selected:
        print(f"{m['id']:<12} {m['category']:<22} {m['status']:<24}")
    print("-" * 60)
    print(f"Total: {len(selected)}")
    n_pass = sum(1 for m in selected if m["currently_pass"])
    n_fail = sum(1 for m in selected if m["currently_pass"] is False)
    n_fixed = sum(1 for m in selected if m["recently_fixed"])
    print(f"Currently PASS: {n_pass}  |  Currently FAIL: {n_fail}  |  Recently fixed: {n_fixed}")
    if warnings:
        print()
        print("Notes:")
        for w in warnings:
            print(f"  - {w}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the review table only; do not write golden_set_30.jsonl",
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    selected, warnings = curate(
        golden_path=args.golden,
        merged_path=args.merged,
        parent_path=args.parent,
    )
    print_review_table(selected, warnings)

    if args.dry_run:
        print("Dry-run: JSONL not written.")
        return 0

    write_jsonl(selected, args.out)
    print(f"Wrote {args.out} ({len(selected)} cases). Approve before treating as final.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
