"""LLMProvider abstraction — PRD.md Section 9, TRD.md Section 4/30.

The application must not depend directly on a single provider, and must not
hardcode assumptions about a specific free model. `FreeLLMAPIProvider` talks
to a local OpenAI-compatible proxy (docs/ADR/0003); `MockProvider` gives
deterministic, dependency-free responses for tests and offline development.

Failure handling per TRD.md Section 30: on any provider error, raise
`LLMUnavailableError` rather than silently substituting a different model or
swallowing the failure. The caller (packages/agent's graph, once built) is
responsible for preserving investigation state and allowing retry —
`packages/analysis` must keep working with no LLM at all.
"""

from __future__ import annotations

from typing import Protocol

import httpx
from pydantic import BaseModel

from packages.domain.db import Settings


class LLMMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class LLMUnavailableError(Exception):
    """The configured provider could not be reached or returned an error.

    Never caught-and-ignored to fall back to a different model — TRD.md
    Section 30: "Do not silently substitute an unknown model."
    """


class LLMProvider(Protocol):
    def complete(
        self, messages: list[LLMMessage], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> LLMResponse: ...


class MockProvider:
    """Deterministic canned response — TRD.md Section 4."""

    def __init__(self, response: str = "This is a mock response.") -> None:
        self._response = response

    def complete(self, messages: list[LLMMessage], *, temperature: float = 0.2, max_tokens: int = 1024) -> LLMResponse:
        return LLMResponse(content=self._response, model="mock", provider="mock", finish_reason="stop")


class FreeLLMAPIProvider:
    """OpenAI-compatible client against a local FreeLLMAPI proxy (docs/ADR/0003).

    Configuration (`base_url`, `api_key`, `model`) is caller-supplied, always
    sourced from env vars upstream (`packages.domain.db.Settings`) — never
    hardcoded here.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not model:
            raise ValueError(
                "LLM_MODEL must be set — FreeLLMAPI exposes many models with no safe default (TRD.md Section 4)"
            )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        # `transport` is a testing hook (httpx.MockTransport) — production
        # callers never pass it.
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def complete(self, messages: list[LLMMessage], *, temperature: float = 0.2, max_tokens: int = 1024) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        try:
            resp = self._client.post(f"{self._base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"FreeLLMAPI request failed: {exc}") from exc

        data = resp.json()
        try:
            choice = data["choices"][0]
            content: str = choice["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMUnavailableError(f"Unexpected FreeLLMAPI response shape: {data}") from exc

        return LLMResponse(
            content=content,
            model=data.get("model", self._model),
            provider="freellmapi",
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
        )


def get_provider(settings: Settings) -> LLMProvider:
    """Factory reading `Settings` — env-var driven, never a hardcoded provider/model."""
    if settings.llm_provider == "mock":
        return MockProvider()
    if settings.llm_provider == "freellmapi":
        return FreeLLMAPIProvider(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model
        )
    raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider!r}")
