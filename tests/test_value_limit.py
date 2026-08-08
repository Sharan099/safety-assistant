"""Value-vs-limit detection, synonym expand, and criteria bias."""

from __future__ import annotations

from retrieval.acronyms import expand_acronyms
from retrieval.retrieve import RetrievedChunk
from retrieval.value_limit import (
    bias_chunks_for_value_vs_limit,
    expand_value_vs_limit_query,
    is_value_vs_limit_query,
)


def test_detect_rdc_pass_fail_case():
    q = (
        "The side impact test produced a Rib Deflection of 45 mm. "
        "Does the vehicle pass UN R95?"
    )
    assert is_value_vs_limit_query(q)
    assert not is_value_vs_limit_query("What is the Rib Deflection Criterion in UN R95?")
    assert not is_value_vs_limit_query("Does the vehicle pass UN R95?")  # no measured value


def test_expand_rib_deflection_synonym_and_rdc():
    q = "Rib Deflection of 45 mm under R95"
    expanded = expand_acronyms(q)
    assert "Rib Deflection Criterion" in expanded
    assert "RDC" in expand_acronyms("What is the RDC limit?")
    assert "Pubic Symphysis Peak Force" in expand_acronyms("PSPF limit")


def test_value_vs_limit_query_adds_injury_criteria_terms():
    q = (
        "The side impact test produced a Rib Deflection of 45 mm. "
        "Does the vehicle pass UN R95?"
    )
    expanded = expand_value_vs_limit_query(q)
    assert "injury" in expanded.lower()
    assert "RDC" in expanded or "Rib Deflection Criterion" in expanded
    assert "42" in expanded  # criterion-specific limit cue


def test_bias_prefers_performance_criteria_over_iso6487():
    limit = RetrievedChunk(
        chunk_id="limit",
        text=(
            "Thorax performance criteria shall be: "
            "(a) Rib Deflection Criterion (RDC) less than or equal to 42 mm;"
        ),
        section_number="5",
        score=0.3,
    )
    calib = RetrievedChunk(
        chunk_id="calib",
        text=(
            "The three thorax rib deflection channels shall comply with "
            "ISO 6487:1987 CFC: 1,000 Hz CAC 60 mm."
        ),
        section_number="Annex 4",
        score=0.9,
    )
    ranked = bias_chunks_for_value_vs_limit([calib, limit], question="RDC 45 mm pass R95?")
    assert ranked[0].chunk_id == "limit"


def test_named_criterion_query_without_numeric():
    from retrieval.value_limit import is_named_criterion_query

    assert is_named_criterion_query("What is the HPC limit in UN R95?")
    assert not is_named_criterion_query("What collision type does R95 address?")
