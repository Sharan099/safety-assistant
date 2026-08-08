"""Context budget + enumerative detection + parent expand caps."""

from __future__ import annotations

from retrieval.context_budget import (
    apply_context_budget,
    approx_tokens,
    is_enumerative_query,
)
from retrieval.expand import _merge_section
from retrieval.retrieve import RetrievedChunk


def _c(cid: str, text: str, **kw) -> RetrievedChunk:
    base = dict(
        chunk_id=cid,
        text=text,
        regulation_id="UN-ECE-R94",
        section_number="5.2.1",
        section_title="HIC",
        section_id=f"UN-ECE-R94::{cid}",
        page_number=12,
        bounding_box=[1.0, 2.0, 3.0, 4.0],
        score=0.5,
    )
    base.update(kw)
    return RetrievedChunk(**base)


def test_enumerative_detection_not_over_triggering():
    assert is_enumerative_query("Generate a checklist for preparing a vehicle for UN R94 testing")
    assert is_enumerative_query("List all frontal impact requirements in R94")
    assert is_enumerative_query("List every requirement related to doors in UN R95")
    assert is_enumerative_query("summarize all requirements for R94")
    assert is_enumerative_query("What are all the requirements for side impact?")
    assert not is_enumerative_query("What is the HIC15 limit in UN R94?")
    assert not is_enumerative_query("What is the ThCC threshold?")


def test_standard_budget_caps_at_5_chunks():
    chunks = [_c(f"c{i}", "word " * 50) for i in range(12)]
    trimmed, stats = apply_context_budget(chunks, question="What is the HIC15 limit?")
    assert stats["mode"] == "standard"
    assert len(trimmed) <= 5
    assert stats["context_chunks_to_llm"] == len(trimmed)
    assert stats["budget_exceeded_incoming"] is True


def test_enum_budget_allows_more_chunks():
    chunks = [_c(f"c{i}", "word " * 20) for i in range(22)]
    trimmed, stats = apply_context_budget(
        chunks, question="Generate a checklist for preparing a vehicle for UN R94 testing"
    )
    assert stats["mode"] == "enumerative"
    assert len(trimmed) <= 20
    assert len(trimmed) >= 8  # higher than standard cap when content fits


def test_merge_section_token_proxy():
    big = _merge_section(
        [_c("a", "alpha " * 100), _c("b", "beta " * 100)],
        score=1.0,
        section_id="UN-ECE-R94::Annex 2",
        cite_from=_c("a", "alpha"),
    )
    assert approx_tokens(big.text) > 100
