"""Groq provider adapter for LLM router."""

from __future__ import annotations

from backend.app.gateway.providers.groq_provider import GroqProvider as _GroqProvider
from backend.app.llm_router.providers.base import CompletionResult, LLMProvider


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str | None = None):
        self._inner = _GroqProvider(api_key=api_key)

    def is_configured(self) -> bool:
        return bool(self._inner.api_key)

    def complete(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float | None = None,
    ) -> CompletionResult:
        return CompletionResult(
            self._inner.complete(
                model_id,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        )

    def list_models(self) -> list[str]:
        return self._inner.list_models()
