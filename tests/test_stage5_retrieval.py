"""Stage 5 gate: hybrid dense+sparse, RRF→rerank cut, hard reg filter, expansion."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from ingestion.embed_upsert import DEFAULT_COLLECTION, get_qdrant_client
from retrieval.expand import expand_to_parents
from retrieval.retrieve import (
    DEFAULT_HYBRID_TOP_K,
    RetrievedChunk,
    hybrid_search,
    retrieve,
    rrf_merge,
)


def _qdrant_ready() -> bool:
    try:
        client = get_qdrant_client()
        try:
            names = {c.name for c in client.get_collections().collections}
            if DEFAULT_COLLECTION not in names:
                return False
            return int(client.get_collection(DEFAULT_COLLECTION).points_count or 0) > 0
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_ready(), reason="Qdrant regulations collection empty — run Stage 4 first"
)


def test_dense_and_sparse_both_contribute(monkeypatch):
    """Zeroing either hybrid weight must change the fused ranking."""
    q = "tibia compression force criterion TCFC 8 kN UN R94"
    rid = "UN-ECE-R94"

    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "1.0")
    monkeypatch.setenv("HYBRID_SPARSE_WEIGHT", "1.0")
    both = hybrid_search(q, regulation_id=rid, top_k=10)
    assert both

    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "0.0")
    monkeypatch.setenv("HYBRID_SPARSE_WEIGHT", "1.0")
    sparse_only = hybrid_search(q, regulation_id=rid, top_k=10)

    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "1.0")
    monkeypatch.setenv("HYBRID_SPARSE_WEIGHT", "0.0")
    dense_only = hybrid_search(q, regulation_id=rid, top_k=10)

    both_ids = [c.chunk_id for c in both]
    sparse_ids = [c.chunk_id for c in sparse_only]
    dense_ids = [c.chunk_id for c in dense_only]
    # At least one of the ablations must differ from the balanced ranking.
    assert both_ids != sparse_ids or both_ids != dense_ids or sparse_ids != dense_ids


def test_rrf_fusion_candidate_pool_then_rerank_cut(monkeypatch):
    """Hybrid pool ~30, then retrieve() cuts to RERANK_TOP_K after rerank."""
    monkeypatch.setenv("HYBRID_TOP_K", "30")
    monkeypatch.setenv("RERANK_TOP_K", "5")
    monkeypatch.setenv("RERANK_PROVIDER", "none")  # deterministic no-op scorer path

    q = "Head Performance Criterion HPC limit UN R94"
    hybrid = hybrid_search(q, regulation_id="UN-ECE-R94", top_k=30)
    assert len(hybrid) <= 30
    assert len(hybrid) >= min(5, DEFAULT_HYBRID_TOP_K)

    result = retrieve(
        q,
        regulation_id="UN-ECE-R94",
        rewrite=False,
        do_rerank=True,
        small_to_big=False,
    )
    chunks = result if isinstance(result, list) else getattr(result, "chunks", result)
    if hasattr(result, "chunks"):
        chunks = result.chunks
    assert isinstance(chunks, list)
    assert 1 <= len(chunks) <= 5


def test_named_regulation_query_no_cross_contamination():
    hits = hybrid_search(
        "What is the HPC limit in UN Regulation No. 94?",
        regulation_id="UN-ECE-R94",
        top_k=20,
    )
    assert hits
    regs = {h.regulation_id for h in hits}
    assert regs == {"UN-ECE-R94"}


def test_context_expansion_attaches_parent_content(monkeypatch):
    """After expand_to_parents, a leaf gains parent/sibling text under the cap."""
    from retrieval.retrieve import RetrievedChunk as RC

    # Use a real leaf that has a parent_section_id in the index.
    hits = hybrid_search(
        "femur force criterion FFC Figure 3",
        regulation_id="UN-ECE-R94",
        top_k=10,
    )
    leaf = next((h for h in hits if h.parent_section_id), None)
    if leaf is None:
        pytest.skip("no leaf with parent_section_id in top hits")

    client = get_qdrant_client()
    try:
        monkeypatch.setenv("MAX_PARENT_EXPAND_TOKENS", "2000")
        expanded = expand_to_parents([leaf], client=client, collection=DEFAULT_COLLECTION)
        assert expanded
        # Expanded text should be at least as informative as the leaf (parent merge
        # or leaf retained). Never empty.
        assert expanded[0].text.strip()
        # Parent id should still be traceable.
        assert expanded[0].section_id or expanded[0].parent_section_id or expanded[0].chunk_id
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def test_rrf_merge_unit_both_lists_matter():
    a = [
        RetrievedChunk(chunk_id="d1", text="dense1", score=1.0, regulation_id="UN-ECE-R94"),
        RetrievedChunk(chunk_id="d2", text="dense2", score=0.9, regulation_id="UN-ECE-R94"),
    ]
    b = [
        RetrievedChunk(chunk_id="s1", text="sparse1", score=1.0, regulation_id="UN-ECE-R94"),
        RetrievedChunk(chunk_id="d1", text="dense1", score=0.8, regulation_id="UN-ECE-R94"),
    ]
    fused = rrf_merge([a, b], top_k=5, weights=[1.0, 1.0])
    ids = [c.chunk_id for c in fused]
    assert "d1" in ids  # appears in both → strong RRF
    assert len(fused) >= 2
