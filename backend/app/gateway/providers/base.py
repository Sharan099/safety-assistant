"""
Provider abstraction.

Every provider speaks the same OpenAI-compatible chat contract internally
(`messages=[{"role", "content"}]`) and returns a normalised `ProviderResponse`.
This is what makes the gateway OpenAI-compatible and multi-provider.
"""

from __future__ import annotations

import abc

from backend.app.gateway.types import ProviderResponse


class ProviderError(RuntimeError):
    """Raised for any provider failure the router may retry / fail over on.

    `retryable` distinguishes transient faults (timeout, 429, 5xx) from
    permanent ones (bad request, auth) so the router can decide quickly.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class Provider(abc.ABC):
    #: stable key used in fallback chains / metrics (e.g. "groq").
    key: str = "base"

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human/metric provider name, e.g. 'groq' or 'anthropic'."""

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """Concrete model id served by this provider instance."""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> ProviderResponse:
        """Run an OpenAI-style chat completion. Raise ProviderError on failure."""

    @abc.abstractmethod
    def available(self) -> bool:
        """True if the provider is configured (keys present, SDK importable)."""
