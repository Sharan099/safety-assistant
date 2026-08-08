"""Multi-criterion retrieval: top-3 per named criterion, then merge."""

from __future__ import annotations

from retrieval.context_budget import context_budgets
from retrieval.multi_criterion import (
    is_multi_criterion_query,
    list_named_criteria,
    merge_per_criterion_chunks,
    uncovered_criteria,
)
from retrieval.retrieve import RetrievedChunk


COMPOSITE_Q = (
    "The test recorded an HPC of 920, chest compression of 32 mm, "
    "and fuel leakage of 40 g/min. Does the vehicle pass?"
)


def test_list_named_criteria_composite():
    named = list_named_criteria(COMPOSITE_Q)
    keys = {c.key for c in named}
    assert keys >= {"hpc", "thcc", "fuel_leakage"}
    assert is_multi_criterion_query(COMPOSITE_Q)
    assert not is_multi_criterion_query("What is the HPC limit in UN R94?")


def test_merge_round_robin_dedupes():
    a = [
        RetrievedChunk(chunk_id="shared", text="HPC and fuel", score=0.9),
        RetrievedChunk(chunk_id="hpc_only", text="HPC", score=0.8),
        RetrievedChunk(chunk_id="hpc_2", text="HPC2", score=0.7),
    ]
    b = [
        RetrievedChunk(chunk_id="shared", text="HPC and fuel", score=0.85),
        RetrievedChunk(chunk_id="fuel_only", text="fuel leakage 30 g/min", score=0.8),
        RetrievedChunk(chunk_id="fuel_2", text="leakage", score=0.7),
    ]
    c = [
        RetrievedChunk(chunk_id="chest", text="chest compression ThCC", score=0.9),
        RetrievedChunk(chunk_id="shared", text="shared", score=0.5),
    ]
    merged = merge_per_criterion_chunks([a, b, c], per_k=3)
    ids = [x.chunk_id for x in merged]
    assert ids[0] == "shared"  # first of first list
    assert "fuel_only" in ids
    assert "chest" in ids
    assert ids.count("shared") == 1


def test_multi_criterion_context_budget():
    chunks, tokens, mode = context_budgets(COMPOSITE_Q)
    assert mode == "multi_criterion"
    assert chunks >= 9  # 3 criteria × top-3


def test_live_composite_retrieval_covers_all_criteria():
    from retrieval.retrieve import retrieve

    chunks = retrieve(
        COMPOSITE_Q,
        regulation_id="UN-ECE-R95",
        top_k=9,
        rewrite=True,
        do_rerank=True,
        small_to_big=False,
    )
    missing = uncovered_criteria(COMPOSITE_Q, chunks)
    assert not missing, f"uncovered={[m.key for m in missing]} chunks={[(c.chunk_id, c.section_number) for c in chunks]}"
    joined = " ".join((c.text or "") for c in chunks).lower()
    assert "hpc" in joined or "head performance" in joined
    assert "fuel" in joined or "leakage" in joined
    assert (
        "chest" in joined
        or "thcc" in joined
        or "thorax" in joined
        or "rib deflection" in joined
    )
