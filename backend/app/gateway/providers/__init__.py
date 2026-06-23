"""Provider registry for the gateway."""

from __future__ import annotations

from backend.app.gateway import config as cfg
from backend.app.gateway.providers.anthropic_provider import (
    AnthropicProvider,
    make_haiku,
    make_sonnet,
)
from backend.app.gateway.providers.base import Provider, ProviderError
from backend.app.gateway.providers.groq_provider import GroqProvider


def build_default_registry() -> dict[str, Provider]:
    """Map provider-key -> Provider instance used by the router/fallback chains."""
    return {
        "groq": GroqProvider(model=cfg.GROQ_TIER_MODEL, key="groq"),
        "groq_power": GroqProvider(model=cfg.GROQ_TIER_MODEL_POWER, key="groq_power"),
        "groq_fast": GroqProvider(model=cfg.GROQ_TIER_MODEL_FAST, key="groq_fast"),
        "anthropic_haiku": make_haiku(),
        "anthropic_sonnet": make_sonnet(),
    }


__all__ = [
    "Provider",
    "ProviderError",
    "GroqProvider",
    "AnthropicProvider",
    "make_haiku",
    "make_sonnet",
    "build_default_registry",
]
