"""APPLICABILITY — per-reg Scope survey with ternary verdicts."""

from __future__ import annotations

from retrieval.applicability import (
    VERDICT_APPLIES,
    VERDICT_DOES_NOT,
    VERDICT_UNKNOWN,
    parse_vehicle_profile,
    render_applicability_answer,
    RegApplicability,
)
from retrieval.router import QueryIntent, classify_query
from retrieval.retrieve import RetrievedChunk


def test_router_x3_ev_applicability():
    q = "Which regulations are applicable for occupant protection of the new BMW X3 EV?"
    routed = classify_query(q, use_llm=False, log=False)
    assert routed.intent == QueryIntent.APPLICABILITY


def test_parse_x3_ev_profile():
    profile = parse_vehicle_profile(
        "Which regulations are applicable for occupant protection of the new BMW X3 EV?"
    )
    assert "M1" in profile.categories
    assert profile.powertrain == "ev"
    assert any("BMW" in m or "X3" in m.upper() for m in profile.model_hints + [profile.raw])


def test_render_lists_all_verdict_sections():
    r94 = RetrievedChunk(
        chunk_id="s94",
        text="This Regulation applies to vehicles of category M1.",
        regulation_id="UN-ECE-R94",
        section_number="1",
        page_number=1,
        score=1.0,
    )
    r95 = RetrievedChunk(
        chunk_id="s95",
        text="This Regulation applies to the lateral collision behaviour of vehicles.",
        regulation_id="UN-ECE-R95",
        section_number="1",
        page_number=1,
        score=1.0,
    )
    decisions = [
        RegApplicability(
            regulation_id="UN-ECE-R94",
            verdict=VERDICT_APPLIES,
            reason="Scope covers M1 frontal impact.",
            scope_chunk=r94,
        ),
        RegApplicability(
            regulation_id="UN-ECE-R95",
            verdict=VERDICT_APPLIES,
            reason="Scope covers side impact.",
            scope_chunk=r95,
        ),
        RegApplicability(
            regulation_id="UN-ECE-R129",
            verdict=VERDICT_DOES_NOT,
            reason="CRS regulation; no CRS ask.",
            scope_chunk=None,
        ),
        RegApplicability(
            regulation_id="UN-ECE-R16",
            verdict=VERDICT_UNKNOWN,
            reason="Need full belt fitment check.",
            scope_chunk=None,
        ),
    ]
    from retrieval.applicability import parse_vehicle_profile

    text, sources = render_applicability_answer(
        decisions,
        vehicle=parse_vehicle_profile("BMW X3 EV occupant protection"),
        to_source=lambda c: c,
    )
    assert "## Applies" in text
    assert "## Does not apply" in text
    assert "Cannot determine" in text
    assert "R94" in text and "R95" in text
    assert len(sources) == 2
