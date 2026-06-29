"""Tests for per-query timing breakdown."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from registry.query_timing import QueryTiming
from registry.search import RegulationSearchEngine


def test_timing_parts_sum_within_tolerance():
    timing = QueryTiming()
    timing.steps = {
        "guardrails_ms": 1.0,
        "dense_retrieval_ms": 100.0,
        "sparse_retrieval_ms": 80.0,
        "rrf_fusion_ms": 2.0,
        "rerank_ms": 0.0,
        "annex_promotion_ms": 1.0,
        "parent_expansion_ms": 5.0,
        "llm_generation_ms": 1200.0,
    }
    total = 1390.0
    out = timing.finalize(total)
    assert abs(out["parts_sum_ms"] + out["overhead_ms"] - out["total_ms"]) <= 50.0
    assert out["slowest_step"] == "llm_generation"


def test_search_returns_timing_field(db_session):
    engine = RegulationSearchEngine()
    fake_chunk = {
        "chunk_id": 1,
        "chunk_text": "ThCC 42 mm",
        "page_number": 1,
        "section": "5.2.1.4",
        "regulation_code": "UN_R94",
        "amendment": "Base",
        "document_name": "UN_R94.pdf",
        "search_score": 0.9,
        "rrf_score": 0.9,
    }
    routing = {"model_key": "test", "model_id": "m", "latency_ms": 10.0, "steps": []}
    with patch.object(engine, "_dense_search_sqlite", return_value=[fake_chunk]):
        with patch.object(engine, "_sparse_search_sqlite", return_value=[]):
            with patch.object(engine.embedder, "embed_query", return_value=[0.0] * 768):
                with patch.object(
                    engine,
                    "_generate_grounded_answer",
                    return_value=("answer", routing),
                ):
                    out = engine.search(
                        db_session,
                        "What is the chest deflection limit under UN R94?",
                        top_k=3,
                        rerank=False,
                    )
    assert "timing" in out
    assert "total_ms" in out["timing"]
    assert out["metadata"]["timing"] == out["timing"]
    delta = abs(
        out["timing"]["parts_sum_ms"] + out["timing"]["overhead_ms"] - out["timing"]["total_ms"]
    )
    assert delta <= 50.0


def test_api_search_timing_endpoint():
    from fastapi.testclient import TestClient

    from api.routes import get_search_engine
    from app.main import app
    from database.connection import get_db

    client = TestClient(app)
    mock_db = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_engine = MagicMock()
    timing = {
        "guardrails_ms": 1.0,
        "dense_retrieval_ms": 50.0,
        "sparse_retrieval_ms": 40.0,
        "rrf_fusion_ms": 1.0,
        "rerank_ms": 0.0,
        "annex_promotion_ms": 0.0,
        "parent_expansion_ms": 2.0,
        "llm_generation_ms": 800.0,
        "rerank_bypassed": True,
        "overhead_ms": 1.0,
        "parts_sum_ms": 894.0,
        "total_ms": 895.0,
        "slowest_step": "llm_generation",
        "slowest_ms": 800.0,
    }
    mock_engine.search.return_value = {
        "answer": "ok",
        "sources": [],
        "timing": timing,
        "metadata": {"timing": timing},
    }
    app.dependency_overrides[get_search_engine] = lambda: mock_engine
    try:
        resp = client.post(
            "/api/v1/search/timing",
            json={"query": "UN R94 chest deflection", "top_k": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["timing"]["slowest_step"] == "llm_generation"
    finally:
        app.dependency_overrides.pop(get_search_engine, None)
        app.dependency_overrides.pop(get_db, None)
