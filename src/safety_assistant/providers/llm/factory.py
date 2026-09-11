from __future__ import annotations

from functools import lru_cache

from safety_assistant.config import Settings, get_settings
from safety_assistant.providers.llm.base import LLMProvider


def build_llm_provider(settings: Settings) -> LLMProvider | None:
    """None means "evidence-only mode": retrieval works, generation is disabled."""
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
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider | None:
    return build_llm_provider(get_settings())
