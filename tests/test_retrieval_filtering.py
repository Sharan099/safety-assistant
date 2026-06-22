"""Retrieval hard-filter integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.hybrid import HybridRetriever  # noqa: E402
from backend.app.retrieval.query_intent import detect_query_intent  # noqa: E402


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()


def test_frontal_query_excludes_side_chunks(retriever):
    result = retriever.retrieve(
        "What is the chest deflection legal limit in frontal impact under UN R94?"
    )
    for d in result["documents"]:
        chunk = retriever._chunk_by_id.get(d["id"], {})
        tt = chunk.get("test_type", "general")
        if tt in ("side", "pole_side", "rear"):
            pytest.fail(f"Side/rear chunk in frontal query: {d['id']} test_type={tt}")


def test_legal_limit_excludes_rating_threshold(retriever):
    result = retriever.retrieve(
        "What is the official legal chest deflection requirement under UN R94?"
    )
    for d in result["documents"]:
        chunk = retriever._chunk_by_id.get(d["id"], {})
        if chunk.get("value_type") == "rating_threshold":
            pytest.fail(f"Rating threshold in legal query: {d['id']}")


def test_intent_detects_frontal():
    intent = detect_query_intent("side impact in Europe under UN R95")
    assert intent.test_type == "side"
    assert intent.region == "EU"


def test_intent_legal_excludes_reference():
    intent = detect_query_intent("What is the legal limit for belt anchorage under UN R14?")
    assert intent.doc_type_intent == "legal"
    assert "reference" in intent.exclude_doc_types
