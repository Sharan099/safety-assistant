"""Enumerative classifier + context-budget interaction."""

from __future__ import annotations

from retrieval.context_budget import apply_context_budget, is_enumerative_query
from retrieval.enumerative import classify_enumerative, detect_named_regulation
from retrieval.retrieve import RetrievedChunk


def test_enumerative_cues_and_non_triggers():
    assert is_enumerative_query("List every requirement related to doors in UN R95")
    assert is_enumerative_query("every requirement related to doors in UN R95")
    assert is_enumerative_query("all requirements related to doors under R95")
    assert is_enumerative_query("list all frontal impact requirements in R94")
    assert is_enumerative_query("summarize all requirements for UN R94")
    assert is_enumerative_query("What are all the requirements for side impact in R95?")
    assert is_enumerative_query("Generate a checklist for preparing a vehicle for UN R94 testing")
    assert not is_enumerative_query("What is the HIC15 limit in UN R94?")
    assert not is_enumerative_query("What is the ThCC threshold?")
    assert not is_enumerative_query("Does the vehicle pass UN R95?")


def test_enumerative_topic_doors():
    from retrieval.enumerative import (
        bias_chunks_for_enumerative_topic,
        extract_enumerative_topic,
        topic_focused_subquery,
    )

    q = "List every requirement related to doors in UN R95"
    assert extract_enumerative_topic(q) == "doors"
    assert "door" in (topic_focused_subquery(q) or "").lower()
    chunks = [
        RetrievedChunk(
            chunk_id="pre",
            text="Application for approval of a vehicle type",
            section_number="3",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="door",
            text="No door shall open during the test. Doors shall be closed but not locked.",
            section_number="5.3.1",
            score=0.4,
        ),
    ]
    ordered = bias_chunks_for_enumerative_topic(chunks, question=q)
    assert ordered[0].chunk_id == "door"


def test_classify_sets_broad_rerank_and_named_reg():
    cls = classify_enumerative("List every requirement related to doors in UN R95")
    assert cls.is_enumerative
    assert cls.rerank_top_k >= 20
    assert cls.hybrid_top_k >= cls.rerank_top_k
    assert cls.named_regulation_id == "UN-ECE-R95"
    assert detect_named_regulation("doors in UN R95") == "UN-ECE-R95"

    narrow = classify_enumerative("What is the HPC limit in UN R94?")
    assert not narrow.is_enumerative
    # Narrow named-reg asks still resolve a hard filter target.
    from retrieval.enumerative import resolve_hard_regulation_filter

    assert resolve_hard_regulation_filter("What is the HPC limit in UN R94?") == "UN-ECE-R94"

def test_enum_budget_allows_up_to_20_chunks():
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            text="word " * 15,
            regulation_id="UN-ECE-R95",
            section_number="5",
            score=0.5,
        )
        for i in range(25)
    ]
    trimmed, stats = apply_context_budget(
        chunks, question="List every requirement related to doors in UN R95"
    )
    assert stats["mode"] == "enumerative"
    assert len(trimmed) <= 20
    assert len(trimmed) >= 8
