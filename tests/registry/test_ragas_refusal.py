"""Tests for RAGAS honest-refusal override."""

from __future__ import annotations

from registry.ragas_refusal import expect_refusal, refusal_is_correct


def test_honesty_case_expects_refusal():
    case = {
        "id": "honesty_nonexistent_reg",
        "expect_refusal": True,
        "ground_truth": "UN R999 is not in the ingested corpus.",
        "question": "What is the chest deflection limit under UN R999?",
    }
    assert expect_refusal(case)
    assert refusal_is_correct(
        case, "I couldn't find any relevant passages in the safety regulation registry."
    )


def test_numeric_case_not_refusal():
    case = {
        "id": "R94",
        "ground_truth": "42 mm",
        "question": "UN R94 chest",
    }
    assert refusal_is_correct(case, "42 mm ThCC") is None
