"""
Anthropic provider (Tier 2 Claude Haiku, Tier 3 Claude Sonnet).

Uses the official `anthropic` SDK. The OpenAI-style `messages` array is converted
to Anthropic's format: any system messages are concatenated into the top-level
`system` parameter, the rest are passed as user/assistant turns.
"""

from __future__ import annotations

import time

from loguru import logger

from backend.app.gateway import config as cfg
from backend.app.gateway.providers.base import Provider, ProviderError
from backend.app.gateway.types import ProviderResponse


class AnthropicProvider(Provider):
    """One instance per model (Haiku or Sonnet)."""

    def __init__(self, model: str, key: str) -> None:
        self._model = model
        self.key = key  # "anthropic_haiku" | "anthropic_sonnet"
        self._client = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> bool:
        if not cfg.ANTHROPIC_API_KEY:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError(
                    "anthropic SDK not installed", retryable=False
                ) from exc
            if not cfg.ANTHROPIC_API_KEY:
                raise ProviderError("ANTHROPIC_API_KEY not set", retryable=False)
            self._client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        return self._client

    @staticmethod
    def _split_messages(
        messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        system_parts: list[str] = []
        turns: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                turns.append({"role": role, "content": content})
        if not turns:
            turns = [{"role": "user", "content": ""}]
        return "\n\n".join(system_parts), turns

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> ProviderResponse:
        t0 = time.perf_counter()
        system, turns = self._split_messages(messages)
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self._model,
                system=system or None,
                messages=turns,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
        except ProviderError:
            raise
        except Exception as exc:
            retryable = _is_retryable(exc)
            logger.warning(f"Anthropic chat failed (retryable={retryable}): {exc}")
            raise ProviderError(str(exc), retryable=retryable) from exc

        answer = _extract_text(resp).strip()
        usage = getattr(resp, "usage", None)
        return ProviderResponse(
            answer=answer,
            model=self._model,
            provider="anthropic",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            prompt_tokens=getattr(usage, "input_tokens", None) if usage else None,
            completion_tokens=(
                getattr(usage, "output_tokens", None) if usage else None
            ),
        )


def _extract_text(resp) -> str:
    """Anthropic returns a list of content blocks; concatenate the text ones."""
    content = getattr(resp, "content", None)
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if any(t in name for t in ("timeout", "connection", "ratelimit", "overloaded",
                               "internalserver", "apistatus")):
        return True
    if any(t in msg for t in ("timeout", "429", "rate limit", "overloaded",
                              "503", "502", "500", "529")):
        return True
    if any(t in msg for t in ("401", "403", "invalid api key", "400",
                              "authentication")):
        return False
    return True


def make_haiku() -> AnthropicProvider:
    return AnthropicProvider(cfg.TIERS[2].model, key="anthropic_haiku")


def make_sonnet() -> AnthropicProvider:
    return AnthropicProvider(cfg.TIERS[3].model, key="anthropic_sonnet")
