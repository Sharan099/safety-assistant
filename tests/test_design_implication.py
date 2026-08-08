"""DESIGN_IMPLICATION — component expansion, per-claim grounding, FACT/INFERENCE."""

from __future__ import annotations

from generation.answer import AnswerSegment
from retrieval.design_implication import (
    CLAIM_FACT,
    CLAIM_INFERENCE,
    expand_design_query,
    keep_grounded_design_segments,
    load_design_components,
    match_component,
    normalize_claim_kind,
    render_design_answer,
)
from retrieval.router import QueryIntent, classify_query
from retrieval.retrieve import RetrievedChunk


def test_component_map_loads_and_matches():
    specs = load_design_components(force=True)
    assert any(s.id == "b_pillar" for s in specs)
    assert match_component("What requirements affect B-Pillar design?").id == "b_pillar"
    assert match_component("What requirements affect the driver's seat design?").id == (
        "drivers_seat"
    )


def test_expand_b_pillar_concepts():
    exp = expand_design_query("What requirements affect B-Pillar design?")
    assert exp.component and exp.component.id == "b_pillar"
    assert any("intrusion" in c.lower() for c in exp.concepts)
    assert "UN-ECE-R95" in exp.likely_regulations
    assert len(exp.subqueries) >= 2


def test_expand_named_r16_vehicle_design():
    exp = expand_design_query(
        "What requirements from UN R16 affect our vehicle design?"
    )
    assert exp.named_regulation_id == "UN-ECE-R16"
    assert exp.component is not None


def test_router_design_intent():
    for q in (
        "What requirements affect B-Pillar design?",
        "What requirements affect the driver's seat design?",
        "What requirements from UN R16 affect our vehicle design?",
    ):
        routed = classify_query(q, use_llm=False, log=False)
        assert routed.intent == QueryIntent.DESIGN_IMPLICATION, q


def test_per_claim_grounding_keeps_valid_drops_bad():
    segs = [
        AnswerSegment(
            text="Door shall not open.",
            citation_chunk_id="good1",
            claim_kind=CLAIM_FACT,
        ),
        AnswerSegment(
            text="Invented claim",
            citation_chunk_id="bad_id",
            claim_kind=CLAIM_FACT,
        ),
        AnswerSegment(
            text="This constrains B-pillar stiffness.",
            citation_chunk_id="good1",
            claim_kind=CLAIM_INFERENCE,
        ),
    ]
    kept, dropped = keep_grounded_design_segments(segs, {"good1"})
    assert len(kept) == 2
    assert dropped == ["bad_id"]
    assert normalize_claim_kind(kept[1].claim_kind) == CLAIM_INFERENCE


def test_render_separates_fact_and_inference():
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="Doors shall not open during the test.",
        regulation_id="UN-ECE-R95",
        section_number="5.3.1",
        page_number=10,
        score=0.9,
    )
    segs = [
        AnswerSegment(
            text="Doors shall not open during the test.",
            citation_chunk_id="c1",
            claim_kind=CLAIM_FACT,
        ),
        AnswerSegment(
            text="This constrains side-structure / B-pillar load path.",
            citation_chunk_id="c1",
            claim_kind=CLAIM_INFERENCE,
        ),
    ]
    text, sources = render_design_answer(
        segs, {"c1": chunk}, to_source=lambda c: c
    )
    assert "Regulatory facts" in text
    assert "Engineering inferences" in text
    assert "[Regulatory fact]" in text
    assert "[Engineering inference]" in text
    assert len(sources) == 1
