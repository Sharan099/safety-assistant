"""LLM gateway — thin facade over the Groq LLM router."""

from __future__ import annotations

from typing import Any

import httpx

from backend.app.gateway import config
from backend.app.gateway.cache import clear_for_tests
from backend.app.gateway.error_policy import reset_disabled_for_tests
from backend.app.gateway.providers.groq_provider import GroqProvider
from backend.app.llm_router.router import LLMRouter, RouterResult, reset_router_for_tests

GatewayResult = RouterResult


class LLMGateway:
    """Thin wrapper — all generation flows through LLMRouter."""

    def __init__(
        self,
        *,
        groq: GroqProvider | None = None,
        primary: str | None = None,
        use_cache: bool = True,
        router: LLMRouter | None = None,
    ):
        if groq is not None:
            from backend.app.llm_router.provider_registry import ProviderRegistry
            from backend.app.llm_router.providers.groq_provider import GroqProvider as RouterGroq

            reg = ProviderRegistry(groq=RouterGroq(api_key=groq.api_key))
            self._router = router or LLMRouter(providers=reg, use_cache=use_cache)
        else:
            self._router = router or LLMRouter(use_cache=use_cache)
        self.groq = groq or GroqProvider()
        self.primary = primary or config.DEFAULT_PRIMARY
        self.use_cache = use_cache

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        context_chunks: list[dict[str, Any]] | None = None,
    ) -> GatewayResult:
        return self._router.complete(
            messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            context_chunks=context_chunks,
        )


def ping_primary_provider() -> None:
    """Startup check: Groq responds."""
    if not config.ENABLE_GATEWAY:
        return
    from backend.app.llm_router.config import api_keys
    from backend.app.llm_router.registry import ordered_chain

    if not api_keys().get("groq"):
        raise RuntimeError("No LLM provider configured — set GROQ_API_KEY")

    chain = ordered_chain()
    if not chain:
        raise RuntimeError("No Groq models in router chain")

    api_key = GroqProvider().api_key
    try:
        with httpx.Client(timeout=10.0, trust_env=True) as client:
            resp = client.get(
                "https://api.groq.com/openai/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "AutoSafety-RAG-Gateway/1.0",
                },
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Gateway provider groq unreachable: {exc}") from exc
    if resp.status_code >= 500:
        raise RuntimeError(f"Gateway provider groq returned HTTP {resp.status_code} at startup")


def reset_gateway_state_for_tests() -> None:
    reset_disabled_for_tests()
    reset_router_for_tests()
    clear_for_tests()
