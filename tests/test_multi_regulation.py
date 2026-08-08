"""Plural / multi-regulation survey retrieval."""

from __future__ import annotations

from retrieval.multi_regulation import (
    chunk_matches_topic,
    format_coverage_summary,
    is_plural_regulation_query,
    topic_terms,
)


def test_plural_regulation_detection():
    assert is_plural_regulation_query(
        "Which regulations include electrical safety requirements?"
    )
    assert is_plural_regulation_query("across all regulations, what covers doors?")
    assert is_plural_regulation_query(
        "What electrical safety requirements apply in general?"
    )
    assert not is_plural_regulation_query("What is the HPC limit in UN R94?")
    assert not is_plural_regulation_query(
        "What vehicles are covered under UN R94?"
    )
    # Named-reg + "in general" stays single-reg (hard filter path).
    assert not is_plural_regulation_query(
        "What is the HPC limit in UN R94 in general?"
    )


def test_topic_relevance_electrical():
    terms = topic_terms("Which regulations include electrical safety requirements?")
    assert "electrical" in terms
    assert "safety" in terms

    class _C:
        def __init__(self, text: str, section_number: str = ""):
            self.text = text
            self.section_number = section_number
            self.section_title = ""

    assert chunk_matches_topic(
        _C("Protection against electrical shock after impact; high voltage bus"),
        terms,
    )
    assert not chunk_matches_topic(
        _C("Safety-belt and restraint systems equipment shall be installed"),
        terms,
    )


def test_coverage_summary_names_missing():
    text = format_coverage_summary(
        covered=["UN-ECE-R94", "UN-ECE-R95"],
        missing=["UN-ECE-R16", "UN-ECE-R129"],
    )
    assert "UN R94" in text and "UN R95" in text
    assert "UN R16" in text and "UN R129" in text
    assert "No relevant content" in text


def test_retrieve_electrical_includes_r94_and_r95():
    from retrieval.retrieve import retrieve

    q = "Which regulations include electrical safety requirements?"
    chunks = retrieve(q, top_k=3, rewrite=False, do_rerank=True, small_to_big=False)
    regs = {c.regulation_id for c in chunks}
    assert "UN-ECE-R94" in regs, regs
    assert "UN-ECE-R95" in regs, regs
    # Off-topic corpora must not leak in via weak hybrid hits.
    assert "UN-ECE-R16" not in regs
