"""
Groq provider (Tier 1).

Wraps the Groq SDK directly so we can pass an OpenAI-style `messages` array and a
per-call timeout (the existing `GroqLLM` builds its own messages and has no
timeout, so the gateway uses the SDK to stay in control of reliability). The
model defaults to the same Groq model the rest of PSA AI already uses.
"""

from __future__ import annotations

import time

from loguru import logger

from backend.app.gateway import config as cfg
from backend.app.gateway.providers.base import Provider, ProviderError
from backend.app.gateway.types import ProviderResponse


class GroqProvider(Provider):
    key = "groq"

    def __init__(self, model: str | None = None) -> None:
        self._model = model or cfg.TIERS[1].model
        self._client = None

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> bool:
        if not cfg.GROQ_API_KEY:
            return False
        try:
            import groq  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise ProviderError(
                    "Groq SDK not installed", retryable=False
                ) from exc
            if not cfg.GROQ_API_KEY:
                raise ProviderError("GROQ_API_KEY not set", retryable=False)
            self._client = Groq(api_key=cfg.GROQ_API_KEY)
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> ProviderResponse:
        t0 = time.perf_counter()
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
        except ProviderError:
            raise
        except Exception as exc:  # SDK raises various transient errors
            retryable = _is_retryable(exc)
            logger.warning(f"Groq chat failed (retryable={retryable}): {exc}")
            raise ProviderError(str(exc), retryable=retryable) from exc

        answer = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        return ProviderResponse(
            answer=answer,
            model=self._model,
            provider="groq",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=(
                getattr(usage, "completion_tokens", None) if usage else None
            ),
        )


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if any(t in name for t in ("timeout", "connection", "ratelimit", "apistatus")):
        return True
    if any(t in msg for t in ("timeout", "429", "rate limit", "503", "502", "500")):
        return True
    # Auth / bad request are permanent.
    if any(t in msg for t in ("401", "403", "invalid api key", "400")):
        return False
    return True
