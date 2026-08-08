"""HPC acronym expand + weighted hybrid RRF / exact-term bias."""

from __future__ import annotations

from retrieval.acronyms import ACRONYMS, expand_acronyms
from retrieval.retrieve import RetrievedChunk, rrf_merge
from retrieval.value_limit import (
    bias_chunks_for_value_vs_limit,
    criteria_focused_subquery,
    exact_regulatory_phrases,
    is_named_criterion_query,
)


HPC_Q = "What is the HPC limit in UN R95?"


def test_hpc_in_acronym_table_and_expands():
    assert "HPC" in ACRONYMS
    assert ACRONYMS["HPC"] == "Head Performance Criterion"
    expanded = expand_acronyms(HPC_Q)
    assert "Head Performance Criterion" in expanded
    assert "HPC (Head Performance Criterion)" in expanded
    # R95 also expands — confirm both fired for the audit case.
    assert "R95 (UN Regulation No. 95" in expanded


def test_named_criterion_triggers_subquery_without_measured_value():
    assert is_named_criterion_query(HPC_Q)
    sq = criteria_focused_subquery(HPC_Q)
    assert sq is not None
    assert "Head Performance Criterion" in sq
    assert "HPC" in exact_regulatory_phrases(HPC_Q)


def test_weighted_rrf_raises_sparse_list():
    dense = [
        RetrievedChunk(chunk_id="elec", text="REESS high voltage", score=0.9),
        RetrievedChunk(chunk_id="hpc", text="Head Performance Criterion (HPC)", score=0.5),
    ]
    sparse = [
        RetrievedChunk(chunk_id="hpc", text="Head Performance Criterion (HPC)", score=20.0),
        RetrievedChunk(chunk_id="elec", text="REESS high voltage", score=5.0),
    ]
    equal = rrf_merge([dense, sparse], top_k=2, weights=[1.0, 1.0])
    boosted = rrf_merge([dense, sparse], top_k=2, weights=[1.0, 1.75])
    assert equal[0].chunk_id in {"hpc", "elec"}
    assert boosted[0].chunk_id == "hpc"


def test_bias_prefers_hpc_clause_over_electrical_safety():
    hpc = RetrievedChunk(
        chunk_id="hpc_limit",
        text=(
            "5.2.1.1. The head performance criterion (HPC) shall be less than "
            "or equal to 1,000; when there is no head contact, then the criterion "
            "is not calculated or is set to zero."
        ),
        section_number="5.2.1.1",
        regulation_id="UN-ECE-R95",
        score=0.4,
    )
    elec = RetrievedChunk(
        chunk_id="elec",
        text=(
            "Annex 9/4 Physical protection Following the vehicle impact test "
            "any parts surrounding the high voltage components shall protect "
            "against electrical shock. REESS coupling system for charging."
        ),
        section_number="Annex 9/4",
        regulation_id="UN-ECE-R95",
        score=0.95,
    )
    ranked = bias_chunks_for_value_vs_limit([elec, hpc], question=HPC_Q)
    assert ranked[0].chunk_id == "hpc_limit"


def test_bias_prefers_named_regulation_hpc():
    r95 = RetrievedChunk(
        chunk_id="r95_hpc",
        text="Head performance criterion (HPC) shall be less than or equal to 1,000",
        section_number="5",
        regulation_id="UN-ECE-R95",
        score=0.5,
    )
    r94 = RetrievedChunk(
        chunk_id="r94_hpc",
        text="The head performance criterion (HPC) shall not exceed 1,000",
        section_number="5.2.1.1",
        regulation_id="UN-ECE-R94",
        score=0.9,
    )
    ranked = bias_chunks_for_value_vs_limit([r94, r95], question=HPC_Q)
    assert ranked[0].chunk_id == "r95_hpc"
