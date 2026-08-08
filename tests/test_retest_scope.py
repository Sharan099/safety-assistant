"""RETEST_SCOPE — modification clauses + mandatory non-authoritative framing."""

from __future__ import annotations

from retrieval.retest_scope import (
    RETEST_DISCLAIMER,
    RETEST_DISCLAIMER_TITLE,
    expand_retest_query,
    match_changes,
    render_retest_answer,
    retest_result_from_chunks,
)
from retrieval.router import QueryIntent, classify_query
from retrieval.retrieve import RetrievedChunk


BMW_Q = (
    "After changing the B-pillar / adding 170kg / relocating the battery, "
    "do we need to retest?"
)


def test_router_bmw_retest_question():
    routed = classify_query(BMW_Q, use_llm=False, log=False)
    assert routed.intent == QueryIntent.RETEST_SCOPE


def test_match_all_three_bmw_changes():
    changes = match_changes(BMW_Q)
    ids = {c.id for c in changes}
    assert "b_pillar" in ids
    assert "mass_increase" in ids
    assert "battery_reess" in ids


def test_expand_targets_passive_safety_regs():
    exp = expand_retest_query(BMW_Q)
    assert {c.id for c in exp.changes} >= {"b_pillar", "mass_increase", "battery_reess"}
    assert exp.named_regulation_id is None
    for ch in exp.changes:
        assert ch.likely_regulations


def test_render_always_carries_disclaimer_and_non_decision():
    chunk = RetrievedChunk(
        chunk_id="mod1",
        text=(
            "Every modification of an approved vehicle type shall be notified "
            "to the Type Approval Authority. Extension of approval may require "
            "further testing."
        ),
        regulation_id="UN-ECE-R94",
        section_number="7",
        page_number=10,
        score=1.0,
    )
    result = retest_result_from_chunks(BMW_Q, [chunk])
    text, sources = render_retest_answer(
        [],
        {"mod1": chunk},
        result=result,
        to_source=lambda c: c,
    )
    assert RETEST_DISCLAIMER_TITLE in text
    assert "homologation authority" in text.lower()
    assert "Informational only" in text or "informational only" in text.lower()
    assert "determination" in text.lower()
    assert "you must retest" not in text.lower()
    assert sources


def test_render_strips_overconfident_model_language():
    from generation.answer import AnswerSegment
    from retrieval.design_implication import CLAIM_INFERENCE

    chunk = RetrievedChunk(
        chunk_id="mod1",
        text="Extension of approval and further testing.",
        regulation_id="UN-ECE-R95",
        section_number="8",
        page_number=12,
        score=1.0,
    )
    result = retest_result_from_chunks(
        "After changing the B-pillar, do we need to retest?",
        [chunk],
    )
    segs = [
        AnswerSegment(
            text="You must retest after this B-pillar change.",
            citation_chunk_id="mod1",
            claim_kind=CLAIM_INFERENCE,
        )
    ]
    text, _ = render_retest_answer(
        segs, {"mod1": chunk}, result=result, to_source=lambda c: c
    )
    assert "you must retest" not in text.lower()
    assert RETEST_DISCLAIMER[:20] in text or "Informational only" in text
