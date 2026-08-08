"""Unit tests for CI regression gate helpers."""

from __future__ import annotations

from eval.gold import load_golden_set
from eval.regression_gate import DEFAULT_CI_CASE_IDS, _is_ci_case


def test_ci_gate_cases_present_in_golden():
    gold = {c["id"]: c for c in load_golden_set()}
    for cid in DEFAULT_CI_CASE_IDS:
        assert cid in gold, f"missing ci-gate case {cid}"
        case = gold[cid]
        assert _is_ci_case(case)
        if cid == "xrg_009":
            assert case.get("regulation_id") == "UN-ECE-R95"
        else:
            # New schema: chunk ids and/or expected_behavior encode pass criteria.
            assert (
                case.get("expected_chunk_ids")
                or case.get("expected_answer_contains")
                or case.get("expected_behavior")
            )


def test_hpoint_expected_chunk():
    case = next(c for c in load_golden_set() if c["id"] == "fac_013")
    assert case["question"] == "Define H-point"
    assert "efaa3fe01ca6c6fd" in case["expected_chunk_ids"]


def test_expand_hpoint_to_spaced_form():
    from retrieval.acronyms import expand_acronyms

    assert "H point" in expand_acronyms("Define H-point")
