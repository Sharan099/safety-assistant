"""Phase 4 — token-aware Groq gateway routing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.gateway.classifier import classify
from backend.app.gateway import config as cfg
from backend.app.gateway.types import RoutingContext


def test_default_models_primary_is_70b():
  assert not (
      cfg.GROQ_TIER_MODEL != cfg.GROQ_TIER_MODEL_POWER
      and "8b" in (cfg.GROQ_TIER_MODEL or "").lower()
  ), "GROQ_TIER_MODEL must not default to 8B — set GROQ_TIER_MODEL=llama-3.3-70b-versatile on HF"


def test_comparison_routes_to_tier2_groq_power():
    ctx = RoutingContext(
        query="How does the frontal impact test in UN R94 differ from FMVSS 208?",
        prompt="context",
        grounding={"confidence": 0.7},
    )
    decision = classify(ctx)
    assert decision.tier >= 2
    assert decision.model == cfg.GROQ_TIER_MODEL_POWER


def test_simple_lookup_uses_primary_groq():
    ctx = RoutingContext(
        query="What anchorage strength test load does UN R14 require?",
        prompt="context",
        grounding={"confidence": 0.8},
        mode="regulation_lookup",
        llm_tier_floor=2,
    )
    decision = classify(ctx)
    assert decision.tier >= 2
    from backend.app.gateway.fallback_safeguards import is_fast_groq_model
    assert not is_fast_groq_model(decision.model)
