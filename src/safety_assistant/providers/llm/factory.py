from __future__ import annotations

from functools import lru_cache

from safety_assistant.config import Settings, get_settings
from safety_assistant.providers.llm.base import LLMProvider


def build_llm_provider(settings: Settings, *, model: str | None = None) -> LLMProvider | None:
    """None means "evidence-only mode": retrieval works, generation is disabled.
    `model` overrides `settings.llm_model` on the same provider (document summaries)."""
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider == "mock":
        if not settings.is_test:
            raise RuntimeError("LLM_PROVIDER=mock is only allowed when APP_ENV=test")
        from safety_assistant.providers.llm.mock import MockLLMProvider

        return MockLLMProvider()
    from safety_assistant.providers.llm.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model or settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider | None:
    return build_llm_provider(get_settings())


def summary_llm_provider(settings: Settings | None = None) -> LLMProvider | None:
    """Provider for document summaries: `summary_model` when set, else the answer model."""
    s = settings or get_settings()
    return build_llm_provider(s, model=s.summary_model or None) if s.summary_model else get_llm_provider()
