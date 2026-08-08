"""Unit tests for RAGAS dashboard aggregations (no matplotlib required for compute)."""

from __future__ import annotations

import math

from eval.render_ragas_dashboard import (
    compute_overall_ragas,
    compute_per_category_ragas,
    _score_color,
)


def test_overall_nanmean_skips_nan_and_reports_coverage():
    cases = [
        {
            "category": "factual_lookup",
            "ragas": {
                "faithfulness": 0.8,
                "answer_relevancy": 0.5,
                "context_precision": float("nan"),
                "context_recall": 0.2,
            },
        },
        {
            "category": "factual_lookup",
            "ragas": {
                "faithfulness": float("nan"),
                "answer_relevancy": 0.7,
                "context_precision": 0.4,
                "context_recall": 0.4,
            },
        },
        {
            "category": "prompt_injection",  # not RAGAS-applicable — ignored by caller
            "ragas": {"faithfulness": 1.0},
        },
    ]
    ragas_only = [c for c in cases if c["category"] != "prompt_injection"]
    overall = compute_overall_ragas(ragas_only)
    assert overall["faithfulness"]["average"] == 0.8
    assert overall["faithfulness"]["n_scored"] == 1
    assert overall["faithfulness"]["n_cases"] == 2
    assert "1/2 cases scored" in overall["faithfulness"]["display"]
    assert overall["answer_relevancy"]["average"] == 0.6
    assert overall["context_precision"]["average"] == 0.4
    assert overall["context_recall"]["average"] == 0.3


def test_per_category_includes_all_ragas_categories():
    cases = [
        {
            "category": "cross_regulation",
            "ragas": {
                "faithfulness": 0.25,
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "context_recall": float("nan"),
            },
        }
    ]
    per = compute_per_category_ragas(cases)
    assert "factual_lookup" in per
    assert "cross_regulation" in per
    assert per["cross_regulation"]["faithfulness"]["average"] == 0.25
    assert per["cross_regulation"]["context_recall"]["average"] is None
    assert per["factual_lookup"]["faithfulness"]["n_cases"] == 0


def test_score_color_bands():
    assert _score_color(0.85) == "#2E7D32"
    assert _score_color(0.5) == "#F9A825"
    assert _score_color(0.49) == "#C62828"
    assert _score_color(None) == "#BDBDBD"
    assert _score_color(float("nan")) == "#BDBDBD"
