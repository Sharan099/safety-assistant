"""Unit tests for RAGAS scorer helpers (no live Qdrant / RAGAS required)."""

from __future__ import annotations

import json
from pathlib import Path

from eval.scoring.ragas_scorer import (
    RAGAS_SCORE_CATEGORIES,
    substring_checks,
)
from generation.llm_client import LLMClient, llm_call_kind_scope


def test_ragas_categories_match_spec():
    assert "factual_lookup" in RAGAS_SCORE_CATEGORIES
    assert "design_implication" in RAGAS_SCORE_CATEGORIES
    assert "prompt_injection" not in RAGAS_SCORE_CATEGORIES


def test_substring_checks_pass_and_fail():
    ok = substring_checks(
        "FAIL: measured 35 g/min exceeds 30 g/min",
        expected_answer_contains=["35", "FAIL"],
        must_not_contain=["3 g/min"],
    )
    assert ok["substring_pass"] is True
    assert ok["missing_expected"] == []
    assert ok["forbidden_hits"] == []

    bad = substring_checks(
        "FAIL at 3 g/min",
        expected_answer_contains=["35", "FAIL"],
        must_not_contain=["3 g/min"],
    )
    assert bad["substring_pass"] is False
    assert "35" in bad["missing_expected"]
    assert "3 g/min" in bad["forbidden_hits"]


def test_substring_checks_thousands_separator_equivalence():
    """fac_003: expected '1000' must match regulation-style '1,000'."""
    answer = (
        "The head performance criterion (HPC) shall not exceed 1,000 and the "
        "resultant head acceleration shall not exceed 80 g for more than 3 ms. "
        "[UN-ECE-R94 §5.2.1.1, p.11]"
    )
    ok = substring_checks(
        answer,
        expected_answer_contains=["1000"],
    )
    assert ok["substring_pass"] is True
    assert ok["missing_expected"] == []

    # Reciprocal: expected with comma, answer without.
    ok2 = substring_checks(
        "HPC shall not exceed 1000.",
        expected_answer_contains=["1,000"],
    )
    assert ok2["substring_pass"] is True

    # Decimal points must stay distinct from thousands stripping.
    assert substring_checks(
        "limit is 1.000 mm",
        expected_answer_contains=["1000"],
    )["substring_pass"] is False


def test_ground_truth_prefers_explicit_reference_chunks():
    from eval.scoring.ragas_scorer import _ground_truth_reference

    gt, refs = _ground_truth_reference(
        {
            "id": "fac_001",
            "expected_chunk_ids": [],
            "expected_answer_contains": ["42"],
            "ground_truth_reference_chunks": [
                "5.2.1.4. The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;"
            ],
        }
    )
    assert "42 mm" in gt
    assert "Reference answer must include" not in gt
    assert refs[0]["text"].startswith("5.2.1.4")


def test_llm_call_kind_logged(tmp_path: Path):
    log_path = tmp_path / "llm_calls.jsonl"
    client = LLMClient(provider="mock", cache_dir=tmp_path / "cache", log_path=log_path)
    with llm_call_kind_scope("system_under_test"):
        client.answer(question="What is HPC?", context="HPC shall not exceed 1000.", chunk_ids=["a"])
    with llm_call_kind_scope("judge"):
        client.judge(messages=[{"role": "user", "content": "score this"}], question="q")

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = {r["role"]: r.get("call_kind") for r in rows}
    assert kinds.get("answer") == "system_under_test"
    assert kinds.get("judge") == "judge"
    for r in rows:
        assert "cost_usd" in r
        assert "input_tokens" in r
        assert "output_tokens" in r
