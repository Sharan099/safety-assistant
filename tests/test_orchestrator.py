"""Orchestrator tests — six nodes, dependency wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.agents.schemas import AGENT_SCHEMAS  # noqa: E402


def _mock_retrieve(docs):
    return {"documents": docs, "latency_ms": 1}


def _legal_doc():
    return {
        "id": "c1",
        "text": "[UN_R94 | Rev | 5]\nChest deflection limit",
        "regulation": "UN_R94",
        "doc_type": "legal",
        "authority_tier": "legal_binding",
        "semantic_score": 0.85,
        "rerank_score": 2.0,
    }


@patch("backend.app.agents.base._generate")
@patch("backend.app.core.services.get_reranker")
@patch("backend.app.core.services.get_retriever")
def test_crew_runs_six_agents(mock_retriever, mock_reranker, mock_gen):
    mock_retriever.return_value = MagicMock()
    mock_retriever.return_value._chunk_by_id = {}
    mock_reranker.return_value.rerank.side_effect = lambda q, docs, **kw: {
        "documents": docs,
        "reranker_used": True,
        "latency_ms": 1,
    }

    def retrieve(query, mode=None):
        return _mock_retrieve([_legal_doc()])

    mock_retriever.return_value.retrieve.side_effect = retrieve

    responses = {
        "simulation": {"status": "ok", "metrics": [{"name": "HIC15", "target": "650", "actual": "580", "unit": "", "status": "pass"}]},
        "regulation": {"status": "ok", "checks": []},
        "root_cause": {"status": "ok", "root_causes": []},
        "knowledge": {"status": "ok", "similar_cases": []},
        "countermeasure": {"status": "ok", "countermeasures": []},
        "program_manager": {"status": "ok", "report_markdown": "# Report", "jira_tickets": []},
    }
    call_idx = {"n": 0}
    agent_order = ["simulation", "regulation", "root_cause", "knowledge", "countermeasure", "program_manager"]

    def fake_gen(prompt, **kwargs):
        name = agent_order[min(call_idx["n"], len(agent_order) - 1)]
        call_idx["n"] += 1
        return {"answer": json.dumps(responses[name])}

    mock_gen.side_effect = fake_gen

    from backend.app.agents.orchestrator import crew

    result = crew.invoke({
        "crash_input": "HIC15|650|580",
        "crash_summary": "frontal",
        "agent_queries": {},
        "agent_outputs": {},
        "citations": [],
        "timing": {},
    })
    assert len(result["agent_outputs"]) == 6
    for name, schema in AGENT_SCHEMAS.items():
        payload = result["agent_outputs"][name]
        if payload.get("status") == "insufficient_data":
            continue
        schema.model_validate(payload)
    assert "report" in result
    assert "summary" in result["report"]
