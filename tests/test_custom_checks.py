"""Exact custom hard-gate checks (no LLM judge)."""

from __future__ import annotations

from eval.scoring.custom_checks import (
    citation_grounding_check,
    cross_regulation_check,
    numeric_verbatim_check,
    regulation_match_check,
    run_custom_hard_gates,
)


FUEL_Q = "If the fuel leakage rate is 35 g/min, does the vehicle pass?"


def test_numeric_verbatim_pass():
    result = numeric_verbatim_check(
        FUEL_Q,
        "FAIL: measured 35 g/min exceeds the 30 g/min limit.",
        must_not_contain=["3 g/min"],
    )
    assert result["pass"] is True
    assert result["missing_verbatim"] == []
    assert result["forbidden_substring_hits"] == []


def test_numeric_verbatim_rejects_truncation():
    result = numeric_verbatim_check(
        FUEL_Q,
        "PASS: measured value of 3 g/min is within limit.",
        must_not_contain=["3 g/min"],
    )
    assert result["pass"] is False
    assert result["missing_verbatim"] or result["forbidden_substring_hits"] or result[
        "forbidden_number_hits"
    ]


def test_citation_grounding_uses_fix4_validator():
    ok = citation_grounding_check(["a", "b"], ["a", "b", "c"])
    assert ok["pass"] is True
    assert ok["invalid_citation_ids"] == []

    bad = citation_grounding_check(["a", "invented"], ["a", "b"])
    assert bad["pass"] is False
    assert "invented" in bad["invalid_citation_ids"]


def test_cross_regulation_blocks_foreign_cite():
    result = cross_regulation_check(
        [
            {"chunk_id": "1", "regulation_id": "UN-ECE-R94"},
            {"chunk_id": "2", "regulation_id": "UN-ECE-R16"},
        ],
        "UN-ECE-R94",
        question="What vehicles are covered under UN R94?",
    )
    assert result["pass"] is False
    assert any(f["regulation_id"] == "UN-ECE-R16" for f in result["foreign_citations"])


def test_cross_regulation_allows_comparative():
    result = cross_regulation_check(
        [
            {"chunk_id": "1", "regulation_id": "UN-ECE-R94"},
            {"chunk_id": "2", "regulation_id": "UN-ECE-R95"},
        ],
        "UN-ECE-R94",
        question="Compare UN R94 and UN R95 in one sentence.",
    )
    assert result["pass"] is True
    assert result["comparative"] is True


def test_cross_regulation_fail_closed_without_regulation_id():
    """Ambiguous citation ownership → BLOCK (not ALLOW)."""
    result = cross_regulation_check(
        ["bare-chunk-id-only"],
        "UN-ECE-R94",
        question="What is the HPC limit in UN R94?",
    )
    assert result["pass"] is False
    assert "fail_closed" in (result.get("skipped_reason") or "")


def test_regulation_match_foreign_requires_decline():
    check = regulation_match_check(
        question="What injury criteria does FMVSS 214 apply to side impact?",
        retrieved_sources=[{"chunk_id": "1", "regulation_id": "UN-ECE-R95"}],
        answer_declined=False,
    )
    assert check["matches"] is False
    assert check["pass"] is False
    assert check["need_decline"] is True

    ok = regulation_match_check(
        question="What injury criteria does FMVSS 214 apply to side impact?",
        retrieved_sources=[{"chunk_id": "1", "regulation_id": "UN-ECE-R95"}],
        answer_declined=True,
    )
    assert ok["pass"] is True


def test_regulation_match_overlap_passes_without_decline():
    check = regulation_match_check(
        question="What is the HPC limit in UN R94?",
        regulation_scope="UN-ECE-R94",
        retrieved_sources=[{"chunk_id": "1", "regulation_id": "UN-ECE-R94"}],
        answer_declined=False,
    )
    assert check["matches"] is True
    assert check["pass"] is True


def test_regulation_match_mismatch_requires_decline():
    check = regulation_match_check(
        question="Summarize frontal HIC limits under UN R95.",
        regulation_scope="UN-ECE-R95",
        retrieved_sources=[{"chunk_id": "1", "regulation_id": "UN-ECE-R94"}],
        answer_declined=False,
    )
    assert check["matches"] is False
    assert check["pass"] is False


def test_hard_gates_for_numeric_safety():
    case = {
        "id": "num_001",
        "category": "numeric_safety",
        "question": FUEL_Q,
        "must_not_contain": ["3 g/min"],
        "regulation_scope": "UN-ECE-R95",
    }
    out = run_custom_hard_gates(
        case,
        answer="FAIL: 35 g/min exceeds 30 g/min.",
        retrieved_chunk_ids=["f4047241b49629e7"],
        cited_sources=[{"chunk_id": "f4047241b49629e7", "regulation_id": "UN-ECE-R95"}],
    )
    assert out["applicable"] is True
    assert out["pass"] is True
    assert out["checks"]["numeric_verbatim"]["pass"] is True
    assert out["checks"]["citation_grounding"]["pass"] is True
    assert out["checks"]["cross_regulation"]["pass"] is True


# Real answers from eval run 20260805T152635Z that previously false-failed
# numeric_verbatim via banned_numbers substring matches (42⊂42.5/42.0, etc.).
_NUM_003_ANSWER = (
    "Overall verdict: FAIL\n"
    "The vehicle did not meet the Rib Deflection Criterion in the UN R95 side "
    "impact test. The measured rib deflection was 42.5 mm, which exceeds the "
    "allowed limit of 42.0 mm. As a result, the vehicle receives an overall "
    "verdict of FAIL.\n\n"
    "- Rib Deflection Criterion: measured 42.5 mm <= limit 42 mm → FAIL "
    "(The measured value of 42.5 mm is greater than the maximum allowed limit "
    "of 42.0 mm, resulting in a FAIL for this criterion.)\n"
    "  source: R95, §5"
)

_NUM_004_ANSWER = (
    "Overall verdict: FAIL\n"
    "The vehicle fails to meet the requirements of UN R94 based on the results "
    "of the frontal impact test. The Head Performance Criterion (HPC) result of "
    "1001.0 exceeds the maximum allowable limit of 1000.0, which is a critical "
    "safety threshold. As a result, the vehicle does not comply with the "
    "regulation.\n\n"
    "- Head Performance Criterion: measured 1001 <= limit 1000 → FAIL "
    "(The measured HPC value of 1001.0 is greater than the allowed limit of "
    "1000.0, resulting in a failure for this criterion.)\n"
    "  source: R94, §5.2.1.1"
)

_NUM_005_ANSWER = (
    "Overall verdict: PASS\n"
    "The side impact Soft Tissue Criterion VC result of 0.95 m/s meets the "
    "requirements of UN R95. Since the measured value is less than or equal to "
    "the specified limit of 1.0 m/s, the criterion is satisfied. As all criteria "
    "are met, the overall verdict is a PASS.\n\n"
    "- Viscous Criterion: measured 0.95 m/s <= limit 1 m/s → PASS "
    "(The measured value of 0.95 m/s is within the acceptable limit of 1.0 m/s, "
    "resulting in a PASS verdict for this criterion.)\n"
    "  source: R95, §5"
)


def test_numeric_verbatim_num_003_no_false_banned_42():
    result = numeric_verbatim_check(
        "Rib Deflection Criterion measured 42.5 mm in the UN R95 side impact. "
        "Does the vehicle pass?",
        _NUM_003_ANSWER,
        must_not_contain=["measured 42 mm", "42 mm is less"],
    )
    assert result["pass"] is True
    assert result["forbidden_number_hits"] == []
    assert result["forbidden_substring_hits"] == []
    assert result["missing_verbatim"] == []


def test_numeric_verbatim_num_004_no_false_banned_1000():
    result = numeric_verbatim_check(
        "The frontal impact test recorded an HPC of 1001. Does the vehicle pass UN R94?",
        _NUM_004_ANSWER,
        must_not_contain=["measured 1000", "HPC of 101"],
    )
    assert result["pass"] is True
    assert result["forbidden_number_hits"] == []
    assert result["forbidden_substring_hits"] == []
    assert result["missing_verbatim"] == []


def test_numeric_verbatim_num_005_no_false_banned_prefix():
    result = numeric_verbatim_check(
        "Side impact Soft Tissue Criterion VC was 0.95 m/s. Does this satisfy UN R95?",
        _NUM_005_ANSWER,
        must_not_contain=["measured 0.9 ", "measured value of 95"],
    )
    assert result["pass"] is True
    assert result["forbidden_number_hits"] == []
    assert result["forbidden_substring_hits"] == []
    assert result["missing_verbatim"] == []


def test_numeric_verbatim_banned_number_is_exact_token_not_substring():
    """Adversarial: banned '1' must not match 10 / 12 / 100 / 1.5."""
    result = numeric_verbatim_check(
        "A reading of 10 mm was recorded. Is that within limit?",
        "PASS: measured 10 mm is within the 12 mm limit (also checked 100 and 1.5).",
        must_not_contain=["1"],
    )
    assert result["pass"] is True
    assert result["forbidden_number_hits"] == []
    assert "1" in result["banned_numbers"]

    bad = numeric_verbatim_check(
        "A reading of 10 mm was recorded. Is that within limit?",
        "PASS: measured value of 1 mm is fine.",
        must_not_contain=["1"],
    )
    assert bad["pass"] is False
    assert "1" in bad["forbidden_number_hits"]


def test_numeric_verbatim_banned_42_does_not_match_decimal_tokens():
    """banned '42' must not match distinct tokens '42.5' or '42.0'."""
    result = numeric_verbatim_check(
        "Rib Deflection Criterion measured 42.5 mm. Does the vehicle pass?",
        "FAIL: measured 42.5 mm exceeds the allowed limit of 42.0 mm.",
        must_not_contain=["42"],
    )
    assert "42" in result["banned_numbers"]
    assert result["forbidden_number_hits"] == []
    assert result["pass"] is True


def test_numeric_verbatim_still_flags_rounded_measurement_phrase():
    result = numeric_verbatim_check(
        "Rib Deflection Criterion measured 42.5 mm in the UN R95 side impact. "
        "Does the vehicle pass?",
        "PASS: measured 42 mm is less than the limit.",
        must_not_contain=["measured 42 mm", "42 mm is less"],
    )
    assert result["pass"] is False
    assert result["forbidden_substring_hits"]
