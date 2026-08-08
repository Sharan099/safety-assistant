"""Extractive factual fallback + per-claim keep for grounding over-rejection."""

from __future__ import annotations

from generation.answer import (
    AnswerSegment,
    StructuredAnswer,
    _extractive_factual_answer,
    keep_grounded_answer_segments,
)
from retrieval.acronyms import expand_acronyms, load_acronym_table
from retrieval.retrieve import RetrievedChunk


def test_keep_grounded_answer_segments_partial():
    allowed = {"good", "also"}
    segs = [
        AnswerSegment(text="ok claim", citation_chunk_id="good"),
        AnswerSegment(text="bad claim", citation_chunk_id="missing"),
        AnswerSegment(text="also ok", citation_chunk_id="also"),
    ]
    kept, dropped = keep_grounded_answer_segments(segs, allowed)
    assert [s.citation_chunk_id for s in kept] == ["good", "also"]
    assert dropped == ["missing"]
    structured = StructuredAnswer(answer_segments=kept)
    assert len(structured.answer_segments) == 2


def test_extractive_factual_hic_matches_hpc_sentence():
    chunks = [
        RetrievedChunk(
            chunk_id="hpc",
            text=(
                "5.2.1.1. The head performance criterion (HPC) shall not exceed "
                "1,000 and the resultant head acceleration shall not exceed 80 g."
            ),
            regulation_id="UN-ECE-R94",
            section_number="5.2.1.1",
            page_number=11,
        )
    ]
    text, sources = _extractive_factual_answer(
        question="What is the HIC15 limit in R94?",
        chunks=chunks,
    )
    assert text
    assert "1,000" in text or "1000" in text
    assert "HPC" in text or "head performance" in text.lower()
    assert sources and sources[0].chunk_id == "hpc"


def test_extractive_factual_hybrid_iii():
    chunks = [
        RetrievedChunk(
            chunk_id="dummy",
            text=(
                "The Hybrid III fiftieth percentile male dummy shall be used for "
                "the frontal impact test described in this Regulation."
            ),
            regulation_id="UN-ECE-R94",
            section_number="5.2",
            page_number=10,
        )
    ]
    text, _sources = _extractive_factual_answer(
        question=(
            "Which anthropomorphic test device is used for UN R94 frontal "
            "occupant protection assessment?"
        ),
        chunks=chunks,
    )
    assert "Hybrid III" in text


def test_extractive_skips_fabricated_hic_amendment_probe():
    chunks = [
        RetrievedChunk(
            chunk_id="hpc",
            text=(
                "When Enhanced Child Restraint Systems are tested, the head "
                "performance criterion (HPC) shall not exceed 1,000."
            ),
            regulation_id="UN-ECE-R129",
            section_number="6.6.4",
        )
    ]
    text, _ = _extractive_factual_answer(
        question=(
            "What HIC36 limit of 700 does UN R129 Amendment 12 Annex 17 "
            "impose for Q10 dummies?"
        ),
        chunks=chunks,
    )
    assert text == ""


def test_extractive_skips_hallucination_probe_style_questions():
    """Must not quote unrelated VC text for a fabricated Soft Tissue Criterion ask."""
    chunks = [
        RetrievedChunk(
            chunk_id="vc",
            text=(
                "5.2.1.5. The viscous criterion (V * C) for the thorax shall not "
                "exceed 1,0 m/s."
            ),
            regulation_id="UN-ECE-R94",
            section_number="5.2.1.5",
        )
    ]
    text, _ = _extractive_factual_answer(
        question=(
            "According to UN R95 clause 5.9.4, is the Soft Tissue Criterion "
            "capped at 0.55 m/s?"
        ),
        chunks=chunks,
    )
    assert text == ""
