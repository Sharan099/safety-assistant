"""Electrolyte / qualitative compliance + retrieval bias."""

from __future__ import annotations

from generation.compliance import (
    evaluate_compliance,
    extract_qualitative_scenarios,
    has_compliance_intent,
    is_compliance_check_query,
)
from retrieval.value_limit import (
    bias_chunks_for_value_vs_limit,
    is_exact_term_boost_query,
    is_named_criterion_query,
)
from retrieval.retrieve import RetrievedChunk


ELECTROLYTE_Q = (
    "REESS remained mounted, but electrolyte entered the passenger compartment. "
    "Does the vehicle comply?"
)


def test_electrolyte_is_named_criterion_and_boost():
    assert is_named_criterion_query(ELECTROLYTE_Q)
    assert is_exact_term_boost_query(ELECTROLYTE_Q)
    assert has_compliance_intent(ELECTROLYTE_Q)
    assert is_compliance_check_query(ELECTROLYTE_Q)


def test_bias_prefers_containment_over_isolation_procedure():
    containment = RetrievedChunk(
        chunk_id="contain",
        text=(
            "5.2.8.2.2 In case of non-aqueous electrolyte REESS there shall be no "
            "liquid electrolyte leakage from the REESS into the passenger compartment"
        ),
        regulation_id="UN-ECE-R94",
        section_number="5.2.8.2.2",
        score=0.5,
    )
    isolation = RetrievedChunk(
        chunk_id="isol",
        text=(
            "Annex 11 isolation resistance measurement method. Fifth step The "
            "electrical isolation value Ri divided by the working voltage."
        ),
        regulation_id="UN-ECE-R94",
        section_number="Annex 11/5.2.2.3.5",
        score=0.9,
    )
    ordered = bias_chunks_for_value_vs_limit(
        [isolation, containment], question=ELECTROLYTE_Q
    )
    assert ordered[0].chunk_id == "contain"


def test_qualitative_electrolyte_fail_overall():
    scenarios = extract_qualitative_scenarios(ELECTROLYTE_Q)
    names = {s.criterion.lower() for s in scenarios}
    assert any("electrolyte" in n for n in names)
    assert any("reess" in n or "retention" in n for n in names)

    result = evaluate_compliance(ELECTROLYTE_Q, regulation_id="UN-ECE-R94")
    assert result is not None
    assert result.overall_verdict == "FAIL"
    elec = next(c for c in result.criteria if "electrolyte" in c.criterion.lower())
    assert elec.verdict == "FAIL"
    text = (result.answer_text or "").upper()
    assert "FAIL" in text
    assert "OVERALL VERDICT" in text or "FAIL" in text


def test_bare_comply_cannot_determine():
    q = "Does the vehicle comply?"
    assert is_compliance_check_query(q)
    result = evaluate_compliance(q)
    assert result is not None
    assert result.overall_verdict == "CANNOT_DETERMINE"
    assert "cannot determine" in (result.answer_text or "").lower()
