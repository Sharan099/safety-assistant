"""
Router: turns a selected provider into an actual completion, applying
per-provider timeouts, bounded retries with exponential backoff + jitter, and
ordered failover across the configured fallback chain.

Fallback chains (config.FALLBACK_CHAINS):
    groq            -> groq, anthropic_haiku, anthropic_sonnet
    anthropic_haiku -> anthropic_haiku, anthropic_sonnet, groq
    anthropic_sonnet-> anthropic_sonnet, anthropic_haiku
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from loguru import logger

from backend.app.gateway import config as cfg
from backend.app.gateway import metrics as gm
from backend.app.gateway.providers.base import Provider, ProviderError
from backend.app.gateway.types import ProviderResponse


@dataclass
class RouteOutcome:
    response: ProviderResponse
    provider_key: str
    fallback_used: bool
    attempts: int
    failover_steps: list[dict] = field(default_factory=list)


class Router:
    def __init__(self, providers: dict[str, Provider]) -> None:
        self._providers = providers

    def _backoff(self, attempt: int) -> float:
        delay = min(cfg.BACKOFF_MAX_S, cfg.BACKOFF_BASE_S * (2 ** attempt))
        return delay + random.uniform(0, cfg.BACKOFF_BASE_S)

    def _call_one(
        self,
        provider: Provider,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        last_exc: Exception | None = None
        for attempt in range(cfg.MAX_RETRIES + 1):
            try:
                return provider.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=cfg.PROVIDER_TIMEOUT_S,
                )
            except ProviderError as exc:
                last_exc = exc
                if not exc.retryable or attempt >= cfg.MAX_RETRIES:
                    raise
                delay = self._backoff(attempt)
                logger.warning(
                    f"{provider.key} attempt {attempt + 1} failed "
                    f"({exc}); retrying in {delay:.2f}s"
                )
                time.sleep(delay)
        raise last_exc or ProviderError("unknown provider error")

    def route(
        self,
        primary_key: str,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        block_fast_tier: bool = False,
    ) -> RouteOutcome:
        chain = cfg.FALLBACK_CHAINS.get(primary_key, [primary_key])
        if block_fast_tier:
            chain = [k for k in chain if k != "groq_fast"] or chain
        attempts = 0
        prev_key = primary_key
        steps: list[dict] = []
        for idx, key in enumerate(chain):
            provider = self._providers.get(key)
            if provider is None or not provider.available():
                steps.append({
                    "provider": key,
                    "model": provider.model if provider else key,
                    "outcome": "skipped_unavailable",
                    "detail": "provider missing or no API key",
                })
                logger.info(f"Skipping unavailable provider '{key}'")
                continue
            try:
                attempts += 1
                resp = self._call_one(
                    provider,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                steps.append({
                    "provider": key,
                    "model": resp.model,
                    "outcome": "success",
                    "detail": "",
                })
                fallback_used = key != primary_key
                if fallback_used:
                    gm.GATEWAY_FALLBACK_TOTAL.labels(
                        from_provider=prev_key, to_provider=key
                    ).inc()
                return RouteOutcome(
                    response=resp,
                    provider_key=key,
                    fallback_used=fallback_used,
                    attempts=attempts,
                    failover_steps=steps,
                )
            except ProviderError as exc:
                detail = str(exc)
                steps.append({
                    "provider": key,
                    "model": provider.model,
                    "outcome": "retryable_error" if exc.retryable else "fatal_error",
                    "detail": detail,
                })
                logger.warning(
                    f"Provider '{key}' ({provider.model}) exhausted: {detail}; "
                    f"failing over to next in chain"
                )
                gm.GATEWAY_FALLBACK_TOTAL.labels(
                    from_provider=key,
                    to_provider=(chain[idx + 1] if idx + 1 < len(chain) else "none"),
                ).inc()
                prev_key = key
                continue

        raise ProviderError(
            f"All providers in fallback chain for '{primary_key}' failed",
            retryable=False,
        )
