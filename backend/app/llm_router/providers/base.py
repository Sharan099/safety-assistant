"""Abstract provider interface for the multi-LLM router."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.gateway.providers.base import ProviderError

__all__ = ["LLMProvider", "ProviderError", "CompletionResult"]


class CompletionResult(dict[str, Any]):
    """Normalized completion payload from any provider."""

    @property
    def text(self) -> str:
        return str(self.get("text") or "")

    @property
    def prompt_tokens(self) -> int:
        return int(self.get("prompt_tokens") or 0)

    @property
    def completion_tokens(self) -> int:
        return int(self.get("completion_tokens") or 0)


class LLMProvider(ABC):
    """Provider plug-in — one implementation per vendor."""

    name: str

    @abstractmethod
    def is_configured(self) -> bool:
        """True when API credentials are present."""

    @abstractmethod
    def complete(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float | None = None,
    ) -> CompletionResult:
        """Run chat completion; raise ProviderError on failure."""

    def list_models(self) -> list[str]:
        """Optional model discovery (NVIDIA)."""
        return []
