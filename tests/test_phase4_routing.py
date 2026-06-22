"""Phase 4 — token-aware Groq gateway routing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.gateway.classifier import classify
from backend.app.gateway import config as cfg
from backend.app.gateway.types import RoutingContext


def test_default_models_not_all_fast():
  assert cfg.GROQ_TIER_MODEL != cfg.GROQ_TIER_MODEL_POWER or "8b" in cfg.GROQ_TIER_MODEL


def test_comparison_routes_to_tier2_groq_power():
    ctx = RoutingContext(
        query="How does the frontal impact test in UN R94 differ from FMVSS 208?",
        prompt="context",
        grounding={"confidence": 0.7},
    )
    decision = classify(ctx)
    assert decision.tier >= 2
    assert decision.model == cfg.GROQ_TIER_MODEL_POWER


def test_simple_lookup_stays_tier1():
    ctx = RoutingContext(
        query="What anchorage strength test load does UN R14 require?",
        prompt="context",
        grounding={"confidence": 0.8},
    )
    decision = classify(ctx)
    assert decision.tier == 1
    assert decision.model == cfg.GROQ_TIER_MODEL
