"""Tests for multi-hop cross-document query decomposition and retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.app.retrieval.multi_hop import is_cross_document_query, retrieve_multi_hop
from backend.app.retrieval.query_decomposition import decompose_cross_document


def test_decompose_cross_document_compliance_query():
    q = "Does our crash report meet the chest-deflection limit in the regulation I uploaded?"
    decomp = decompose_cross_document(q)
    assert decomp.is_cross_document
    assert len(decomp.hops) == 2
    assert decomp.hops[0].authority_role == "binding"
    assert decomp.hops[1].authority_role == "measured"
    assert "legal" in decomp.hops[0].target_doc_types


def test_is_cross_document_query():
    q = "Does our crash report meet the chest deflection limit in the regulation?"
    assert is_cross_document_query(q)


def test_multi_hop_abstention_on_empty_hop(tmp_path: Path):
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(
        json.dumps({"chunks": [{"chunk_id": "c1", "text": "only legal", "tier_confirmed": True}]}),
        encoding="utf-8",
    )
    emb_file = tmp_path / "emb.json"
    emb_file.write_text(json.dumps({"embeddings": {}}), encoding="utf-8")

    from backend.app.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(chunks_file=chunks_file, embeddings_file=emb_file)
    q = "Does our crash report meet the chest-deflection limit in the regulation I uploaded?"
    result = retrieve_multi_hop(retriever, q)
    assert result["multi_hop"]["any_abstain"] is True
    assert result["intent_flags"] == ["multi_hop"]
