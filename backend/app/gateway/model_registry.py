"""Canonical model registry — re-exports from the Groq LLM router."""

from __future__ import annotations

from backend.app.llm_router.config import RouterConfig
from backend.app.llm_router.registry import ModelSpec, REGISTRY, get_spec, ordered_chain


def ordered_keys(primary: str) -> list[str]:
    """Failover order from config (primary kept for API compatibility)."""
    _ = primary
    return [s.key for s in ordered_chain(RouterConfig.from_env())]


__all__ = ["ModelSpec", "REGISTRY", "get_spec", "ordered_keys", "ordered_chain"]
