"""Provider factory — Groq only."""

from __future__ import annotations

from backend.app.llm_router.providers.base import LLMProvider
from backend.app.llm_router.providers.groq_provider import GroqProvider


class ProviderRegistry:
    def __init__(self, *, groq: LLMProvider | None = None):
        self._providers: dict[str, LLMProvider] = {
            "groq": groq or GroqProvider(),
        }

    def get(self, provider_name: str) -> LLMProvider:
        p = self._providers.get(provider_name)
        if p is None:
            raise KeyError(f"Unknown provider: {provider_name}")
        return p

    def set(self, provider_name: str, provider: LLMProvider) -> None:
        self._providers[provider_name] = provider
