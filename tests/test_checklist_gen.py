"""CHECKLIST_GEN — per-category retrieval + structured cited checklist."""

from __future__ import annotations

from generation.answer import AnswerSegment
from retrieval.checklist import (
    expand_checklist_query,
    is_checklist_pipeline_query,
    keep_grounded_checklist_segments,
    load_checklist_categories,
    render_checklist_answer,
)
from retrieval.router import QueryIntent, classify_query
from retrieval.retrieve import RetrievedChunk


def test_checklist_pipeline_cues():
    assert is_checklist_pipeline_query(
        "Generate a checklist for preparing a vehicle for UN R94 testing"
    )
    assert is_checklist_pipeline_query(
        "Generate a checklist for preparing a vehicle for UN R94 homologation"
    )
    assert not is_checklist_pipeline_query(
        "List every requirement related to doors in UN R95"
    )


def test_categories_load():
    cats = load_checklist_categories(force=True)
    ids = {c.id for c in cats}
    assert {
        "vehicle_prep",
        "test_prep",
        "dummy_installation",
        "instrumentation",
        "injury_criteria",
        "documentation",
    } <= ids


def test_expand_resolves_r94():
    exp = expand_checklist_query(
        "Generate a checklist for preparing a vehicle for UN R94 testing"
    )
    assert exp.regulation_id == "UN-ECE-R94"
    assert len(exp.categories) >= 6


def test_router_checklist_intent():
    routed = classify_query(
        "Generate a checklist for preparing a vehicle for UN R94 testing",
        use_llm=False,
        log=False,
    )
    assert routed.intent == QueryIntent.CHECKLIST_GEN


def test_per_item_grounding():
    segs = [
        AnswerSegment(
            text="Set fuel to …",
            citation_chunk_id="good",
            category_id="vehicle_prep",
        ),
        AnswerSegment(
            text="Invented",
            citation_chunk_id="bad",
            category_id="vehicle_prep",
        ),
    ]
    kept, dropped = keep_grounded_checklist_segments(segs, {"good"})
    assert len(kept) == 1
    assert dropped == ["bad"]


def test_render_notes_missing_categories():
    from retrieval.checklist import expand_checklist_query

    exp = expand_checklist_query(
        "Generate a checklist for preparing a vehicle for UN R94 testing"
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="The vehicle shall be …",
        regulation_id="UN-ECE-R94",
        section_number="1",
        page_number=2,
        score=0.9,
    )
    segs = [
        AnswerSegment(
            text="Prepare the vehicle per procedure.",
            citation_chunk_id="c1",
            category_id="vehicle_prep",
        )
    ]
    text, sources = render_checklist_answer(
        segs,
        {"c1": chunk},
        expansion=exp,
        covered=["vehicle_prep"],
        missing=["documentation", "instrumentation"],
        chunk_category={"c1": "vehicle_prep"},
        to_source=lambda c: c,
    )
    assert "## Vehicle preparation" in text
    assert "- [ ]" in text
    assert "No content found" in text or "incomplete" in text.lower()
    assert "Documentation" in text or "documentation" in text.lower()
    assert len(sources) == 1
