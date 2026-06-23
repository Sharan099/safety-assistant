"""Aggregate gateway tier routing from eval answer cache."""

from __future__ import annotations

from typing import Any

from backend.app.gateway.fallback_safeguards import is_fast_groq_model


def aggregate_gateway_tier_stats(cache_items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Compute primary vs fast-tier routing distribution from cached generation results.
    A silent shift to 100% 8B fallback should surface here as fast_tier_rate → 1.0.
    """
    total = 0
    fast = 0
    fallback = 0
    models: dict[str, int] = {}

    for entry in cache_items.values():
        model = entry.get("served_model") or ""
        if not model:
            gw = entry.get("gateway") or {}
            rd = entry.get("routing_diagnostic") or gw.get("routing_diagnostic") or {}
            model = rd.get("served_model") or gw.get("model") or ""
        if not model:
            continue
        total += 1
        models[model] = models.get(model, 0) + 1
        if is_fast_groq_model(model):
            fast += 1
        if entry.get("fallback_used") or (entry.get("gateway") or {}).get("fallback_used"):
            fallback += 1

    rate = fast / max(total, 1)
    return {
        "samples_with_model": total,
        "fast_tier_count": fast,
        "fast_tier_rate": round(rate, 4),
        "fallback_count": fallback,
        "fallback_rate": round(fallback / max(total, 1), 4),
        "model_distribution": models,
        "gate_pass": rate < 0.5 if total else True,
    }
