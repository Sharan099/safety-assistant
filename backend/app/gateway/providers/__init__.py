"""Provider registry for the gateway."""

from __future__ import annotations

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
        "groq": GroqProvider(),
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
