"""Unit tests for eval.smoke_subset (no live LLM / retrieval calls)."""

from __future__ import annotations

import pytest

from eval.smoke_subset import (
    confirm_smoke_or_abort,
    print_smoke_summary,
    select_smoke_cases,
)


def _case(cid: str, category: str) -> dict:
    return {"id": cid, "category": category, "question": f"q for {cid}"}


FULL_SET = [
    _case("fac_001", "factual_lookup"),
    _case("fac_002", "factual_lookup"),
    _case("cmp_001", "compliance_check"),
    _case("num_001", "numeric_safety"),
    _case("enm_001", "enumerative"),
    _case("mhp_001", "multi_hop"),
    _case("pin_001", "prompt_injection"),
    _case("hlp_001", "hallucination_probe"),
]


def test_select_smoke_cases_picks_one_per_required_slot():
    sel = select_smoke_cases(FULL_SET)
    cats = [c["category"] for c in sel]
    assert len(sel) == 5
    assert len(set(c["id"] for c in sel)) == 5  # all distinct
    assert "factual_lookup" in cats
    assert "compliance_check" in cats
    assert "numeric_safety" in cats
    assert ("enumerative" in cats) or ("multi_hop" in cats)
    assert ("prompt_injection" in cats) or ("hallucination_probe" in cats)


def test_select_smoke_cases_prefers_first_category_in_group():
    # enumerative should win over multi_hop, prompt_injection over hallucination_probe.
    sel = select_smoke_cases(FULL_SET)
    cats = [c["category"] for c in sel]
    assert "enumerative" in cats
    assert "multi_hop" not in cats
    assert "prompt_injection" in cats
    assert "hallucination_probe" not in cats


def test_select_smoke_cases_falls_back_to_second_choice_in_group():
    without_enumerative = [c for c in FULL_SET if c["category"] != "enumerative"]
    sel = select_smoke_cases(without_enumerative)
    cats = [c["category"] for c in sel]
    assert "multi_hop" in cats


def test_select_smoke_cases_raises_when_required_category_missing():
    incomplete = [c for c in FULL_SET if c["category"] != "compliance_check"]
    with pytest.raises(RuntimeError, match="compliance_check"):
        select_smoke_cases(incomplete)


def test_select_smoke_cases_raises_when_critical_group_missing():
    incomplete = [
        c for c in FULL_SET if c["category"] not in {"prompt_injection", "hallucination_probe"}
    ]
    with pytest.raises(RuntimeError, match="prompt_injection or hallucination_probe"):
        select_smoke_cases(incomplete)


def _summary(all_passed: bool, cases: list[dict]) -> dict:
    n_pass = sum(1 for c in cases if c.get("pass"))
    return {
        "n_cases": len(cases),
        "n_pass": n_pass,
        "all_passed": all_passed,
        "cases": cases,
        "wall_clock_seconds": 12.3,
        "total_cost_usd": 0.01,
        "total_tokens": 500,
        "total_input_tokens": 400,
        "total_output_tokens": 100,
    }


def test_confirm_smoke_or_abort_true_when_all_passed():
    summary = _summary(True, [{"id": "a", "category": "factual_lookup", "pass": True}])
    assert confirm_smoke_or_abort(summary, assume_yes=False) is True


def test_confirm_smoke_or_abort_assume_yes_bypasses_failure():
    summary = _summary(
        False, [{"id": "a", "category": "numeric_safety", "pass": False, "error": "boom"}]
    )
    assert confirm_smoke_or_abort(summary, assume_yes=True) is True


def test_confirm_smoke_or_abort_non_interactive_aborts(monkeypatch):
    import sys

    summary = _summary(False, [{"id": "a", "category": "numeric_safety", "pass": False}])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert confirm_smoke_or_abort(summary, assume_yes=False) is False


def test_print_smoke_summary_does_not_raise(capsys):
    summary = _summary(
        True,
        [
            {"id": "a", "category": "factual_lookup", "pass": True, "_smoke_timing_seconds": 1.2},
            {"id": "b", "category": "numeric_safety", "pass": True, "_smoke_timing_seconds": 0.9},
        ],
    )
    print_smoke_summary(summary)
    out = capsys.readouterr().out
    assert "SMOKE SUBSET" in out
    assert "factual_lookup" in out
