"""Numeric hallucination guard: user 35 g/min must not become 3 g/min."""

from __future__ import annotations

import json
from pathlib import Path

from generation.answer import FAILURE_NUMERIC_HALLUCINATION, answer_question
from generation.llm_client import LLMClient, LLMResult
from generation.numeric_guard import (
    NUMERIC_CONFIRM_MESSAGE,
    check_numeric_fidelity,
    extract_user_numbers,
    log_numeric_hallucination,
)
from retrieval.retrieve import RetrievedChunk


FUEL_Q = "If the fuel leakage rate is 35 g/min, does the vehicle pass?"


def test_extract_user_numbers_fuel_and_variants():
    nums = extract_user_numbers(FUEL_Q)
    assert any(u.normalized == "35" and u.unit == "g/min" for u in nums)

    assert any(
        u.normalized == "42.5" and u.unit == "mm"
        for u in extract_user_numbers(
            "Rib Deflection Criterion measured 42.5 mm in UN R95. Pass?"
        )
    )
    assert any(
        u.normalized == "1001"
        for u in extract_user_numbers(
            "The frontal impact test recorded an HPC of 1001. Does the vehicle pass UN R94?"
        )
    )
    assert any(
        u.normalized == "0.95" and u.unit == "m/s"
        for u in extract_user_numbers(
            "Side impact Soft Tissue Criterion VC was 0.95 m/s. Satisfy UN R95?"
        )
    )
    assert any(
        u.normalized == "7.5" and u.unit == "kn"
        for u in extract_user_numbers(
            "Pubic Symphysis Peak Force of 7.5 kN under UN R95 — comply?"
        )
    )


def test_reject_35_becomes_3_gmin():
    bad = (
        "The measured value of 3 g/min is less than the limit, so the vehicle passes."
    )
    result = check_numeric_fidelity(FUEL_Q, bad)
    assert result.rejected
    assert result.offending_token in {"3", "3.0"}


def test_accept_faithful_fuel_fail():
    good = (
        "FAIL. Fuel-feed leakage shall not exceed 30 g/min; "
        "measured 35 g/min exceeds the limit (35 > 30)."
    )
    result = check_numeric_fidelity(FUEL_Q, good)
    assert result.ok
    assert "35" in good
    assert "3 g/min" not in good


def test_reject_omitted_measured_figure_with_verdict():
    # Verdict without restating 35 at all.
    bad = "PASS. The leakage rate is within the 30 g/min limit."
    result = check_numeric_fidelity(FUEL_Q, bad)
    assert result.rejected


def test_reject_digit_corruption_variants():
    cases = [
        (
            "Rib Deflection Criterion measured 42.5 mm in the UN R95 side impact. Does the vehicle pass?",
            "FAIL. Limit is 42 mm; measured value of 42 mm exceeds the limit.",
        ),
        (
            "The frontal impact test recorded an HPC of 1001. Does the vehicle pass UN R94?",
            "FAIL. HPC shall not exceed 1000; the measured value of 100 exceeds the limit.",
        ),
        (
            "Side impact Soft Tissue Criterion VC was 0.95 m/s. Does this satisfy UN R95?",
            "PASS. Limit is 1.0 m/s; measured value of 0.9 m/s is within the limit.",
        ),
    ]
    for question, bad_answer in cases:
        result = check_numeric_fidelity(question, bad_answer)
        assert result.rejected, (question, bad_answer, result.reason)


def test_log_numeric_hallucination_writes_jsonl(tmp_path: Path):
    path = tmp_path / "numeric_hallucination.jsonl"
    bad = "The measured value of 3 g/min is less than the limit, so the vehicle passes."
    result = check_numeric_fidelity(FUEL_Q, bad)
    assert result.rejected
    log_numeric_hallucination(
        question=FUEL_Q,
        answer=bad,
        result=result,
        trace_id="t-test",
        path=path,
    )
    line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
    record = json.loads(line)
    assert record["severity"] == "CRITICAL"
    assert record["incident"] == "numeric_hallucination"
    assert record["offending_token"] in {"3", "3.0"}


def test_answer_path_rejects_hallucinated_measurement(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "generation.numeric_guard.NUMERIC_HALLUCINATION_PATH",
        tmp_path / "numeric_hallucination.jsonl",
    )
    # Force the legacy LLM path so we still exercise the post-gen numeric guard
    # (compliance-check queries now short-circuit before the LLM).
    monkeypatch.setattr(
        "generation.compliance.is_compliance_check_query",
        lambda *_a, **_k: False,
    )

    chunk = RetrievedChunk(
        chunk_id="fuel1",
        text=(
            "If there is continuous leakage of liquid from the fuel-feed installation "
            "after the collision, the rate of leakage shall not exceed 30 g/min."
        ),
        regulation_id="UN-ECE-R95",
        section_number="5.3.6",
        page_number=10,
        bounding_box=[0, 0, 1, 1],
        score=0.9,
    )

    def fake_complete(self, *args, **kwargs):
        return LLMResult(
            text=json.dumps(
                {
                    "answer_segments": [
                        {
                            "text": (
                                "The measured value of 3 g/min is less than the limit, "
                                "so the vehicle passes"
                            ),
                            "citation_chunk_id": "fuel1",
                        }
                    ]
                }
            ),
            model="mock",
            provider="mock",
            role="answer",
            input_tokens=1,
            output_tokens=1,
        )

    monkeypatch.setattr(LLMClient, "complete", fake_complete)
    llm = LLMClient(
        provider="mock",
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm.jsonl",
        use_cache=False,
    )
    result = answer_question(
        FUEL_Q,
        chunks=[chunk],
        llm=llm,
        skip_answer_cache=True,
        regulation_id="UN-ECE-R95",
    )
    assert result.failure_kind == FAILURE_NUMERIC_HALLUCINATION
    assert result.not_found is True
    assert NUMERIC_CONFIRM_MESSAGE in result.answer
    assert "passes" not in result.answer.lower() or "confirm" in result.answer.lower()
    assert (tmp_path / "numeric_hallucination.jsonl").exists()


def test_compliance_path_uses_exact_35_not_hallucinated_3():
    """Deterministic compliance must never invert 35 g/min → 3 g/min."""
    from generation.compliance import evaluate_compliance
    from ingestion.extract_limits import seed_known_limits

    seed_known_limits()
    result = evaluate_compliance(FUEL_Q, regulation_id="UN-ECE-R95")
    assert result is not None
    assert result.overall_verdict == "FAIL"
    assert any(c.measured == 35 and c.verdict == "FAIL" for c in result.criteria)
    assert "3 g/min" not in result.answer_text
    assert "35" in result.answer_text

