"""Unit tests for crash-development crew agents."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.agents.registry import build_regulation_agent  # noqa: E402
from backend.app.agents.schemas import AGENT_SCHEMAS  # noqa: E402
from backend.app.retrieval.citations import detect_authority_blur_flags  # noqa: E402


def test_regulation_agent_returns_json_or_insufficient_data():
    agent = build_regulation_agent()
    state = {
        "crash_input": "Chest Deflection target 34mm actual 42mm",
        "crash_summary": "frontal chest",
        "agent_queries": {},
        "agent_outputs": {},
        "citations": [],
    }
    out = agent(state)
    reg = out["agent_outputs"]["regulation"]
    assert isinstance(reg, dict)
    assert reg.get("status") in ("ok", "insufficient_data")
    if reg.get("status") == "ok":
        AGENT_SCHEMAS["regulation"].model_validate(reg)
    else:
        assert reg == {"status": "insufficient_data"}
    # Must not be free prose at top level
    assert "checks" in reg or reg.get("status") == "insufficient_data"


def test_regulation_agent_no_tier_blur_on_sample():
    """Advisory tiers must not be phrased as binding compliance."""
    citations = [
        {
            "marker": "S1",
            "authority_tier": "rating_protocol",
            "authority_tier_badge": "RATING",
            "is_legal": False,
        }
    ]
    bad_answer = json.dumps({
        "status": "ok",
        "checks": [{
            "metric": "HIC",
            "body": "Euro NCAP",
            "regulation_id": "Euro NCAP",
            "limit": "must comply",
            "result": "fail",
            "source": "[S1]",
        }],
    })
    flags = detect_authority_blur_flags(bad_answer, citations)
    assert flags, "expected authority blur flag for binding language on RATING source"


@patch("backend.app.agents.base._generate")
@patch("backend.app.core.services.get_reranker")
@patch("backend.app.core.services.get_retriever")
def test_regulation_agent_mocked_json(mock_retriever, mock_reranker, mock_gen):
    mock_retriever.return_value = MagicMock()
    mock_retriever.return_value._chunk_by_id = {}
    mock_retriever.return_value.retrieve.return_value = {
        "documents": [{
            "id": "c1",
            "text": "[UN_R94 | Rev | 5.2.3]\nHIC limit 1000",
            "regulation": "UN_R94",
            "doc_type": "legal",
            "authority_tier": "legal_binding",
            "semantic_score": 0.9,
        }],
        "latency_ms": 1,
    }
    mock_reranker.return_value.rerank.return_value = {
        "documents": mock_retriever.return_value.retrieve.return_value["documents"],
        "reranker_used": True,
        "latency_ms": 1,
    }
    mock_gen.return_value = {
        "answer": json.dumps({
            "status": "ok",
            "checks": [{
                "metric": "HIC15",
                "body": "UN-ECE",
                "regulation_id": "UN R94",
                "limit": "1000",
                "result": "pass",
                "source": "[S1]",
            }],
        }),
    }
    agent = build_regulation_agent()
    out = agent({
        "crash_input": "HIC15|650|580",
        "crash_summary": "frontal",
        "agent_queries": {},
        "agent_outputs": {},
        "citations": [],
    })
    reg = out["agent_outputs"]["regulation"]
    assert reg["status"] == "ok"
    assert reg["checks"][0]["metric"] == "HIC15"
