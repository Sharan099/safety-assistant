from __future__ import annotations

import logging
from functools import lru_cache

from safety_assistant.config import Settings, get_settings
from safety_assistant.providers.llm.base import LLMProvider

log = logging.getLogger(__name__)


def build_llm_provider(
    settings: Settings, *, model: str | None = None, base_url: str | None = None, api_key: str | None = None
) -> LLMProvider | None:
    """None means "evidence-only mode": retrieval works, generation is disabled.
    `model` / `base_url` / `api_key` override the primary settings (document summaries, fallback route)."""
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider == "mock":
        if not settings.is_test:
            raise RuntimeError("LLM_PROVIDER=mock is only allowed when APP_ENV=test")
        from safety_assistant.providers.llm.mock import MockLLMProvider

        return MockLLMProvider()
    from safety_assistant.providers.llm.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        base_url=base_url or settings.llm_base_url,
        api_key=api_key or settings.llm_api_key,
        model=model or settings.llm_model,
        timeout=settings.llm_timeout_seconds,
        reasoning_effort=settings.llm_reasoning_effort or None,
    )


class FallbackLLM:
    """Primary model first; on a provider failure or malformed output, one attempt on the fallback
    model. The response records which model actually answered. Worst case is two call budgets."""

    name = "openai_compatible"

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self.primary, self.fallback = primary, fallback
        self.model = primary.model

    def generate(self, messages, *, schema=None, temperature=0.0, max_tokens=1024):  # type: ignore[no-untyped-def]
        from safety_assistant.observability import metrics
        from safety_assistant.providers.llm.base import LLMError

        try:
            return self.primary.generate(messages, schema=schema, temperature=temperature, max_tokens=max_tokens)
        except LLMError as exc:
            metrics.LLM_CALLS.labels(provider=self.name, outcome=f"fallback:{type(exc).__name__}").inc()
            log.warning(
                "primary model %s failed (%s); trying %s", self.primary.model, type(exc).__name__, self.fallback.model
            )
            return self.fallback.generate(messages, schema=schema, temperature=temperature, max_tokens=max_tokens)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider | None:
    s = get_settings()
    primary = build_llm_provider(s)
    if primary is None or not s.llm_fallback_model or s.llm_fallback_model == s.llm_model:
        return primary
    fallback = build_llm_provider(
        s, model=s.llm_fallback_model, base_url=s.llm_fallback_base_url or None, api_key=s.llm_fallback_api_key or None
    )
    assert fallback is not None
    return FallbackLLM(primary, fallback)


def summary_llm_provider(settings: Settings | None = None) -> LLMProvider | None:
    """Provider for document summaries: `summary_model` when set, else the answer model."""
    s = settings or get_settings()
    return build_llm_provider(s, model=s.summary_model or None) if s.summary_model else get_llm_provider()
