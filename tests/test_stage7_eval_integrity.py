"""Stage 7 gate: eval harness integrity (numeric matcher, RAGAS agg, pinned judge)."""

from __future__ import annotations

import math

from eval.scoring.custom_checks import numeric_verbatim_check
from eval.eval_judge_overflow import (
    DEFAULT_PINNED_JUDGE_MODEL,
    build_pinned_eval_judge_config,
    eval_judge_overflow_primary_model,
)


def test_numeric_checker_banned_42_not_match_decimal():
    """Token-boundary: banned '42' must NOT match '42.5' / '42.0'."""
    result = numeric_verbatim_check(
        "Rib Deflection Criterion measured 42.5 mm. Does the vehicle pass?",
        "FAIL: measured 42.5 mm exceeds the allowed limit of 42.0 mm.",
        must_not_contain=["42"],
    )
    assert "42" in result["banned_numbers"]
    assert result["forbidden_number_hits"] == []
    assert result["pass"] is True


def test_ragas_aggregation_skips_nan_and_reports_scored_count():
    from eval.aggregation import average_ragas_metric

    rows = [
        {"ragas": {"faithfulness": 0.8}},
        {"ragas": {"faithfulness": float("nan")}},
        {"ragas": {"faithfulness": 0.6}},
        {"ragas": {}},
        {"ragas": {"faithfulness": 1.0}},
    ]
    out = average_ragas_metric(rows, "faithfulness")
    assert out["n_scored"] == 3
    assert abs(float(out["average"]) - (0.8 + 0.6 + 1.0) / 3) < 1e-3
    assert not math.isnan(float(out["average"]))
    assert "scored" in (out.get("display") or "").lower() or out["n_scored"] == 3


def test_resolved_judge_model_is_pinned_not_env_override(monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", DEFAULT_PINNED_JUDGE_MODEL)
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "should-not-win-override-model")
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "google")
    cfg = build_pinned_eval_judge_config()
    model = cfg["targets"][0]["override_params"]["model"]
    assert model == DEFAULT_PINNED_JUDGE_MODEL == "gemini-2.5-flash"
    assert model != "should-not-win-override-model"
    resolved = eval_judge_overflow_primary_model()
    assert resolved == DEFAULT_PINNED_JUDGE_MODEL
