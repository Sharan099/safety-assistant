"""Tests for eval.render_dashboard (no live eval required)."""

from __future__ import annotations

import json
from pathlib import Path

from eval.render_dashboard import (
    DEFAULT_CRITICAL_CATEGORIES,
    FIG_DPI,
    FIG_HEIGHT_IN,
    FIG_WIDTH_IN,
    PENDING_BAR_COLOR,
    _bar_color,
    _category_rows,
    _critical_categories,
    _pass_rate_pct,
    _status_color,
    build_results_from_partial,
    render_dashboard,
)


def _sample_results() -> dict:
    return {
        "run_id": "test_dash",
        "timestamp": "2026-08-05T12:00:00+00:00",
        "n_cases": 42,
        "overall_status": "NEEDS IMPROVEMENT",
        "per_category": {
            "factual_lookup": {"pass_rate": 0.95, "n_cases": 20},
            "numeric_safety": {"pass_rate": 1.0, "n_cases": 6},
            "prompt_injection": {"pass_rate": 0.55, "n_cases": 10},
            "multi_hop": {"pass_rate": 0.8, "n_cases": 6},
        },
        "gate": {
            "overall_status": "NEEDS IMPROVEMENT",
            "categories": {
                "factual_lookup": {"severity": "medium"},
                "numeric_safety": {"severity": "critical"},
                "prompt_injection": {"severity": "critical"},
                "multi_hop": {"severity": "medium"},
            },
        },
        "cost_summary": {
            "total_cost_usd": 1.2345,
            "total_tokens": 123456,
            "wall_clock_seconds": 372.5,
        },
    }


def test_bar_color_thresholds():
    assert _bar_color(90) == "#2E7D32"
    assert _bar_color(89.9) == "#F9A825"
    assert _bar_color(70) == "#F9A825"
    assert _bar_color(69.9) == "#C62828"
    assert _bar_color(None, pending=True) == PENDING_BAR_COLOR
    assert _bar_color(0.0, pending=True) == PENDING_BAR_COLOR


def test_status_colors():
    assert _status_color("PRODUCTION READY") == "#2E7D32"
    assert _status_color("NOT PRODUCTION READY") == "#C62828"
    assert _status_color("NEEDS IMPROVEMENT") == "#F9A825"
    assert _status_color("PARTIAL — 4/150 (not a final verdict)").startswith("#")


def test_critical_includes_defaults_and_gate():
    results = _sample_results()
    crit = _critical_categories(results)
    assert "numeric_safety" in crit
    assert "prompt_injection" in crit
    assert DEFAULT_CRITICAL_CATEGORIES <= crit


def test_pass_rate_pct():
    assert _pass_rate_pct({"pass_rate": 0.85}) == 85.0
    assert _pass_rate_pct({"pass_rate": 85}) == 85.0
    assert _pass_rate_pct({"pending": True, "n_cases": 0, "pass_rate": None}) is None
    assert _pass_rate_pct({"n_cases": 0, "pass_rate": None}) is None


def test_category_rows_mark_critical():
    rows = _category_rows(_sample_results())
    by_name = {r[0]: r for r in rows}
    assert by_name["numeric_safety"][2] is True
    assert by_name["prompt_injection"][2] is True
    assert by_name["factual_lookup"][2] is False
    # Sorted ascending by pass rate (evaluated only)
    assert rows[0][0] == "prompt_injection"
    assert all(len(r) == 4 for r in rows)


def test_category_rows_pending_grey_not_zero():
    results = {
        "per_category": {
            "numeric_safety": {"pass_rate": 1.0, "n_cases": 1},
            "prompt_injection": {"n_cases": 0, "n_pass": 0, "pass_rate": None, "pending": True},
            "factual_lookup": {"n_cases": 0, "pass_rate": None, "pending": True},
        },
        "gate": {
            "categories": {
                "numeric_safety": {"severity": "critical"},
                "prompt_injection": {"severity": "critical"},
            }
        },
    }
    rows = _category_rows(results)
    by_name = {r[0]: r for r in rows}
    assert by_name["numeric_safety"][1] == 100.0
    assert by_name["numeric_safety"][3] is False
    assert by_name["prompt_injection"][1] is None
    assert by_name["prompt_injection"][3] is True
    assert by_name["factual_lookup"][3] is True
    # Pending categories sorted after evaluated
    assert rows[0][0] == "numeric_safety"
    assert rows[-1][3] is True


def test_render_dashboard_png(tmp_path: Path):
    out = tmp_path / "dashboard.png"
    path = render_dashboard(_sample_results(), out_path=out)
    assert path.is_file()
    assert path.stat().st_size > 1000

    # Verify slide pixel size without requiring Pillow.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    arr = mpimg.imread(path)
    # imread returns (H, W, C)
    assert arr.shape[1] == int(FIG_WIDTH_IN * FIG_DPI)
    assert arr.shape[0] == int(FIG_HEIGHT_IN * FIG_DPI)


def test_render_partial_dashboard_banner(tmp_path: Path):
    results = {
        "run_id": "partial_test",
        "timestamp": "2026-08-05T12:00:00+00:00",
        "partial": True,
        "n_cases": 4,
        "n_cases_completed": 4,
        "n_cases_total": 150,
        "overall_status": "PARTIAL — 4/150 (not a final verdict)",
        "per_category": {
            "numeric_safety": {"pass_rate": 1.0, "n_cases": 1, "n_pass": 1},
            "prompt_injection": {
                "n_cases": 0,
                "n_pass": 0,
                "pass_rate": None,
                "pending": True,
            },
        },
        "gate": {
            "categories": {
                "numeric_safety": {"severity": "critical"},
                "prompt_injection": {"severity": "critical"},
            }
        },
        "cost_summary": {"total_cost_usd": 0.0, "total_tokens": 10, "wall_clock_seconds": 1.0},
    }
    out = tmp_path / "partial_dashboard.png"
    path = render_dashboard(results, out_path=out)
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_build_results_from_partial(tmp_path: Path):
    run_dir = tmp_path / "20260805T999999Z"
    run_dir.mkdir()
    rows = [
        {"id": "num_001", "category": "numeric_safety", "pass": True, "_case_cost": {"cost_usd": 0.01, "input_tokens": 10, "output_tokens": 5}},
        {"id": "cmp_001", "category": "compliance_check", "pass": True, "_case_cost": {"cost_usd": 0.02, "input_tokens": 20, "output_tokens": 5}},
    ]
    (run_dir / "partial_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": "20260805T999999Z",
                "case_ids": ["num_001", "cmp_001", "pin_001", "fac_001"],
                "gold_path": str(Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"),
                "thresholds_path": str(
                    Path(__file__).resolve().parents[1] / "eval" / "thresholds.yaml"
                ),
            }
        ),
        encoding="utf-8",
    )
    payload = build_results_from_partial(run_dir)
    assert payload["partial"] is True
    assert payload["n_cases_completed"] == 2
    assert payload["n_cases_total"] == 4
    assert payload["per_category"]["numeric_safety"]["pass_rate"] == 1.0
    assert payload["per_category"]["prompt_injection"]["pending"] is True
    assert payload["per_category"]["prompt_injection"]["pass_rate"] is None
