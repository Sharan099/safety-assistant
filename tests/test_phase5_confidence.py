"""Phase 5 — grounding confidence bands."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.citations import assess_grounding, confidence_band


def test_confidence_bands():
    assert confidence_band(0.8) == "high"
    assert confidence_band(0.55) == "medium"
    assert confidence_band(0.2) == "low"


def test_assess_grounding_includes_band():
    docs = [{"semantic_score": 0.8, "rerank_score": 2.0}]
    g = assess_grounding(docs, reranker_used=True, min_semantic=0.45, min_rerank_prob=0.5)
    assert g["confidence_band"] in ("high", "medium", "low")
    assert g["confidence_band"] == "high"
