"""Regression: build_prompt must return state when context exists."""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.app.graph.workflow import RAGWorkflow


def _sample_doc() -> dict:
    return {
        "id": "chunk-1",
        "text": "Chest deflection limits for frontal impact testing.",
        "regulation": "UN_R94",
        "heading_path": "UN_R94 > 5.2.1",
        "title": "5.2.1 Chest",
        "semantic_score": 0.72,
        "rerank_score": 2.5,
    }


def test_build_prompt_returns_context_and_prompt_when_docs_present():
    wf = RAGWorkflow(MagicMock(), MagicMock())
    state = {
        "query": "Which requirements apply to chest deflection in frontal crash?",
        "documents": [_sample_doc()],
        "metadata": {"reranker_used": True},
        "timing": {},
    }
    out = wf._node_build_prompt(state)
    assert out is not None
    assert out.get("context")
    assert out.get("prompt")
    assert out.get("grounding")
    assert not out["metadata"].get("abstain")


def test_build_prompt_abstains_when_no_documents():
    wf = RAGWorkflow(MagicMock(), MagicMock())
    state = {
        "query": "anything",
        "documents": [],
        "metadata": {},
        "timing": {},
    }
    out = wf._node_build_prompt(state)
    assert out is not None
    assert not out.get("context")
    assert out["metadata"].get("abstain")
