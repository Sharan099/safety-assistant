"""Repeat-run retrieval stability (Fix 12 confirmation)."""

from __future__ import annotations

import os

import pytest

from eval.determinism_eval import _chunk_fingerprint, assert_determinism, run_determinism_check


def test_chunk_fingerprint_stable_ordering():
    class C:
        def __init__(self, chunk_id: str) -> None:
            self.chunk_id = chunk_id

    assert _chunk_fingerprint([C("a"), C("b")]) == ("a", "b")


def test_repeat_retrieve_same_chunks_standard_query():
    """Same factual question must retrieve the same ordered chunk ids."""
    # Skip cleanly when the local vector store is unavailable.
    try:
        from retrieval.retrieve import indexed_regulation_ids

        regs = indexed_regulation_ids()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"qdrant unavailable: {exc}")
    if "UN-ECE-R94" not in regs:
        pytest.skip("UN-ECE-R94 not indexed")

    os.environ.setdefault("LLM_PROVIDER", "mock")
    os.environ["RETRIEVAL_REWRITE_LLM"] = "0"
    os.environ.setdefault("QDRANT_EXACT_SEARCH", "1")

    report = run_determinism_check(
        question="What is the HIC15 limit in UN R94?",
        n=3,
        regulation_id="UN-ECE-R94",
        use_llm_rewrite=False,
    )
    assert_determinism(report)
    assert report["chunks_identical"] is True
    assert len(report["canonical_chunk_ids"]) >= 1
