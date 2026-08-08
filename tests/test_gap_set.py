"""Tests for gap-set computation and golden-framed merge."""

from __future__ import annotations

import pytest

from eval.aggregation import (
    GapMergeError,
    compute_gap_set,
    merge_fill_gaps_results,
)


def test_compute_gap_set_missing_failed_and_error():
    gold = [
        {"id": "a", "category": "factual_lookup"},
        {"id": "b", "category": "factual_lookup"},
        {"id": "c", "category": "design_implication"},
        {"id": "d", "category": "guardrail"},
        {"id": "e", "category": "cross_regulation"},
    ]
    results = [
        {"id": "a", "category": "factual_lookup", "pass": True},
        {"id": "b", "category": "factual_lookup", "pass": False},
        {
            "id": "c",
            "category": "design_implication",
            "pass": False,
            "error": "Rate limit (HTTP 429)",
            "answer": None,
        },
        {"id": "d", "category": "guardrail", "pass": True},
        # e missing
    ]
    gap = compute_gap_set(gold, results)
    assert gap["n_golden"] == 5
    assert gap["n_gap"] == 3
    assert gap["n_passed"] == 2
    assert gap["n_missing"] == 1
    assert gap["n_failed"] == 1
    assert gap["n_error"] == 1
    assert gap["missing_ids"] == ["e"]
    assert gap["failed_ids"] == ["b"]
    assert gap["error_ids"] == ["c"]
    assert set(gap["gap_ids"]) == {"b", "c", "e"}
    assert gap["by_category"]["factual_lookup"] == 1
    assert gap["by_category"]["design_implication"] == 1
    assert gap["by_category"]["cross_regulation"] == 1
    assert len(gap["gap_cases"]) == 3


def test_compute_gap_set_empty_when_all_pass():
    gold = [{"id": "a", "category": "factual_lookup"}]
    results = [{"id": "a", "category": "factual_lookup", "pass": True}]
    gap = compute_gap_set(gold, results)
    assert gap["n_gap"] == 0
    assert gap["gap_cases"] == []


def test_merge_fill_gaps_keeps_passers_and_swaps_gap():
    gold = [
        {"id": "a", "category": "factual_lookup"},
        {"id": "b", "category": "factual_lookup"},
        {"id": "c", "category": "cross_regulation"},
    ]
    parent = [
        {"id": "a", "pass": True, "answer": "old-a"},
        {"id": "b", "pass": False, "answer": "old-b"},
        {"id": "c", "pass": True, "answer": "old-c"},
    ]
    new_gap = {
        "b": {"id": "b", "pass": True, "answer": "new-b"},
    }
    merged = merge_fill_gaps_results(
        gold,
        parent_cases=parent,
        gap_ids=["b"],
        new_gap_results=new_gap,
    )
    assert merged["n_merged"] == 3
    assert merged["n_golden"] == 3
    assert [r["id"] for r in merged["cases"]] == ["a", "b", "c"]
    assert merged["cases"][0]["answer"] == "old-a"
    assert merged["cases"][1]["answer"] == "new-b"
    assert merged["cases"][2]["answer"] == "old-c"
    assert merged["kept_ids"] == ["a", "c"]
    assert merged["rescored_ids"] == ["b"]


def test_merge_fill_gaps_fails_on_count_mismatch():
    gold = [
        {"id": "a", "category": "factual_lookup"},
        {"id": "b", "category": "factual_lookup"},
    ]
    parent = [{"id": "a", "pass": True}]
    with pytest.raises(GapMergeError, match="missing from parent"):
        merge_fill_gaps_results(
            gold,
            parent_cases=parent,
            gap_ids=[],
            new_gap_results={},
        )


def test_merge_fill_gaps_fails_when_gap_result_missing():
    gold = [
        {"id": "a", "category": "factual_lookup"},
        {"id": "b", "category": "factual_lookup"},
    ]
    parent = [
        {"id": "a", "pass": True},
        {"id": "b", "pass": False},
    ]
    with pytest.raises(GapMergeError, match="new results missing"):
        merge_fill_gaps_results(
            gold,
            parent_cases=parent,
            gap_ids=["b"],
            new_gap_results={},
        )
