"""Tests for hybrid RRF merge, rewrite heuristics, small-to-big dedupe."""

from __future__ import annotations

from retrieval.expand import _merge_section
from retrieval.retrieve import RetrievedChunk, rrf_merge
from retrieval.rewrite import rewrite_query


def _c(cid: str, score: float = 0.0, **kw) -> RetrievedChunk:
    base = dict(
        chunk_id=cid,
        text=f"text-{cid}",
        regulation_id="UN-ECE-R94",
        section_number="5.2.1",
        section_title="HIC",
        section_id=f"UN-ECE-R94::{cid}",
        page_number=12,
        bounding_box=[1.0, 2.0, 3.0, 4.0],
        score=score,
    )
    base.update(kw)
    return RetrievedChunk(**base)


def test_rrf_merge_prefers_cross_list_agreement():
    a = [_c("x", 0.9), _c("y", 0.8), _c("z", 0.7)]
    b = [_c("z", 0.95), _c("x", 0.5), _c("w", 0.4)]
    merged = rrf_merge([a, b], top_k=3)
    ids = [c.chunk_id for c in merged]
    # x and z appear in both lists → should dominate.
    assert ids[0] in {"x", "z"}
    assert "x" in ids and "z" in ids
    assert len(merged) == 3


def test_rewrite_expands_hic_and_splits():
    result = rewrite_query(
        "What is the HIC15 limit in R94 and also what is the TTI limit?",
        use_llm=False,
    )
    assert "Head Injury Criterion" in result.expanded
    assert result.original.startswith("What is the HIC15")
    assert len(result.subqueries) >= 2


def test_rewrite_expands_vc_locally():
    from retrieval.acronyms import expand_acronyms

    q = "What is the VC limit?"
    assert expand_acronyms(q) == "What is the VC (Viscous Criterion) limit?"
    result = rewrite_query(q, use_llm=False)
    assert result.original == q
    assert "Viscous Criterion" in result.expanded
    assert result.expanded == "What is the VC (Viscous Criterion) limit?"


def test_extract_acronyms_from_r94_style_text():
    from retrieval.acronyms import extract_acronyms_from_text

    text = (
        "The head performance criterion (HPC) shall not exceed 1,000. "
        "The Thorax Compression Criterion (ThCC) shall not exceed 42 mm."
    )
    found = extract_acronyms_from_text(text)
    assert "head performance" in found["HPC"].lower()
    assert "ThCC" in found


def test_scope_objective_classifier():
    from retrieval.retrieve import is_definition_query, is_scope_objective_query

    assert is_scope_objective_query("What is the objective of this regulation?")
    assert is_scope_objective_query("What is the scope of UN R94?")
    assert is_scope_objective_query("What is this regulation about?")
    assert not is_scope_objective_query("What is the VC limit?")
    assert not is_scope_objective_query("Define H-point")
    assert is_definition_query("Define H-point")
    assert is_definition_query("What is the definition of protective system?")
    assert not is_definition_query("What is the VC limit?")


def test_prepend_scope_chunks_orders_first():
    from retrieval.retrieve import prepend_scope_chunks

    scope = [_c("scope", section_number="1", section_title="Scope")]
    hybrid = [_c("a"), _c("scope"), _c("b")]
    out = prepend_scope_chunks(scope, hybrid)
    assert [c.chunk_id for c in out] == ["scope", "a", "b"]


def test_merge_section_dedupes_text():
    kids = [
        _c("a", section_id="UN-ECE-R94::5.2", parent_section_id="UN-ECE-R94::5.2", text="Alpha"),
        _c("b", section_id="UN-ECE-R94::5.2.1", parent_section_id="UN-ECE-R94::5.2", text="Beta"),
        _c("c", section_id="UN-ECE-R94::5.2.1", parent_section_id="UN-ECE-R94::5.2", text="Beta"),
    ]
    merged = _merge_section(kids, score=0.5, section_id="UN-ECE-R94::5.2")
    assert merged.text.count("Beta") == 1
    assert "Alpha" in merged.text
    assert merged.chunk_id.startswith("expanded::")


def test_rerank_none_preserves_order():
    from retrieval.rerank import Reranker, rerank

    chunks = [_c("a", 0.1), _c("b", 0.2), _c("c", 0.3)]
    out = rerank("q", chunks, top_n=2, reranker=Reranker(provider="none"))
    assert [c.chunk_id for c in out] == ["a", "b"]
