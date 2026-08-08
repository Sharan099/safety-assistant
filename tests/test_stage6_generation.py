"""Stage 6 gate: numeric safety, citation grounding, absence-claim guard."""

from __future__ import annotations

from generation.answer import AnswerSegment, StructuredAnswer, keep_grounded_answer_segments
from generation.compliance import evaluate_compliance
from generation.numeric_guard import check_numeric_fidelity
from generation.semantic_grounding import (
    filter_unsupported_negative_segments,
    heuristic_claim_supported,
    is_negative_or_absence_claim,
)
from retrieval.retrieve import RetrievedChunk


FUEL_Q = "If the fuel leakage rate is 35 g/min, does the vehicle pass?"


def test_fuel_35_gmin_verbatim_yields_fail_never_3():
    """Digit-hallucination regression: 35 g/min must stay verbatim → FAIL."""
    good = (
        "Overall verdict: FAIL\n"
        "Fuel-feed leakage rate: measured 35 g/min <= limit 30 g/min → FAIL"
    )
    fidelity = check_numeric_fidelity(FUEL_Q, good)
    assert fidelity.ok
    assert "35 g/min" in good
    assert "3 g/min" not in good

    bad = "Overall verdict: PASS. Measured 3 g/min is within the 30 g/min limit."
    assert check_numeric_fidelity(FUEL_Q, bad).rejected

    result = evaluate_compliance(FUEL_Q, regulation_id="UN-ECE-R94")
    assert result is not None
    assert result.overall_verdict == "FAIL"
    # Measured value preserved — never rewritten to 3.
    measured_vals = [
        str(c.measured) for c in (result.criteria or []) if c.measured is not None
    ]
    assert any(v.startswith("35") for v in measured_vals) or "35" in (
        result.summary or ""
    )


def test_citation_to_non_retrieved_chunk_rejected():
    allowed = {"retrieved_a", "retrieved_b"}
    segs = [
        AnswerSegment(text="Grounded claim", citation_chunk_id="retrieved_a"),
        AnswerSegment(text="Hallucinated cite", citation_chunk_id="not_retrieved_xyz"),
    ]
    kept, dropped = keep_grounded_answer_segments(segs, allowed)
    assert [s.citation_chunk_id for s in kept] == ["retrieved_a"]
    assert "not_retrieved_xyz" in dropped
    structured = StructuredAnswer(answer_segments=kept)
    assert all(s.citation_chunk_id in allowed for s in structured.answer_segments)


def test_absence_claim_not_emitted_unless_context_states_it():
    claim = "There is no relationship between UN R16 and UN R94."
    assert is_negative_or_absence_claim(claim)

    silent_chunk = RetrievedChunk(
        chunk_id="c1",
        text=(
            "5.2.1.1. The head performance criterion (HPC) shall not exceed 1,000.\n"
            "5.2.1.4. The Thorax Compression Criterion (ThCC) shall not exceed 42 mm."
        ),
        regulation_id="UN-ECE-R94",
        section_number="5.2.1.1",
    )
    ok, reason = heuristic_claim_supported(claim, silent_chunk.text)
    assert ok is False
    assert "without_explicit_absence" in reason

    segs = [AnswerSegment(text=claim, citation_chunk_id="c1")]
    kept, dropped = filter_unsupported_negative_segments(
        segs, {"c1": silent_chunk}
    )
    assert dropped
    assert kept == []

    explicit = RetrievedChunk(
        chunk_id="c2",
        text="This regulation does not apply to vehicles of category L.",
        regulation_id="UN-ECE-R94",
        section_number="1",
    )
    ok2, reason2 = heuristic_claim_supported(
        "This requirement does not apply to category L vehicles.",
        explicit.text,
    )
    assert ok2 is True
