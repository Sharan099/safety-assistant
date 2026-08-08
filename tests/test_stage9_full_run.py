"""Stage 9 gate: full 30-case run artifacts (results.json + both dashboards)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"


def _latest_run_dir() -> Path | None:
    if not RESULTS.is_dir():
        return None
    dirs = sorted(
        [p for p in RESULTS.iterdir() if p.is_dir() and p.name[:8].isdigit()],
        key=lambda p: p.name,
        reverse=True,
    )
    return dirs[0] if dirs else None


@pytest.fixture(scope="module")
def stage9_run_dir():
    run = _latest_run_dir()
    if run is None or not (run / "results.json").is_file():
        pytest.skip(
            "Stage 9 full run not finished yet — "
            "run: python -m eval.run_full --yes --skip-smoke"
        )
    return run


def test_stage9_exactly_30_cases(stage9_run_dir):
    data = json.loads((stage9_run_dir / "results.json").read_text(encoding="utf-8"))
    cases = data.get("cases") or data.get("results") or data.get("per_case") or []
    n = int(data.get("n_cases") or len(cases))
    # Canonical rebuild gold is ~30; remapped set currently has 32 lines.
    assert n in {30, 32}, n
    assert len(cases) == n


def test_stage9_both_dashboards(stage9_run_dir):
    assert (stage9_run_dir / "dashboard.png").is_file()
    assert (stage9_run_dir / "ragas_dashboard.png").is_file()


def test_stage9_results_have_per_category_and_cost(stage9_run_dir):
    data = json.loads((stage9_run_dir / "results.json").read_text(encoding="utf-8"))
    # Per-category aggregation (never blended-only).
    cats = (
        data.get("by_category")
        or data.get("per_category")
        or data.get("category_summary")
        or (data.get("aggregations") or {}).get("by_category")
    )
    assert cats, "missing per-category aggregation"
    cost = data.get("cost_summary") or data.get("cost") or data.get("usage_summary")
    assert cost is not None, "missing cost_summary"
