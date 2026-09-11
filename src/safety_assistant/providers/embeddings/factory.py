from __future__ import annotations

from functools import lru_cache

from safety_assistant.config import EMBEDDING_DIMENSIONS, Settings, get_settings
from safety_assistant.providers.embeddings.base import EmbeddingProvider


class ProviderConfigurationError(RuntimeError):
    pass


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hashing":
        if not settings.is_test:
            raise ProviderConfigurationError("EMBEDDING_PROVIDER=hashing is only allowed when APP_ENV=test")
        from safety_assistant.providers.embeddings.hashing import HashingEmbeddingProvider

        return HashingEmbeddingProvider(EMBEDDING_DIMENSIONS)
    from safety_assistant.providers.embeddings.fastembed import FastEmbedProvider

    provider = FastEmbedProvider(settings.embedding_model)
    if provider.dimensions != EMBEDDING_DIMENSIONS:
        raise ProviderConfigurationError(
            f"{settings.embedding_model} produces {provider.dimensions}-d vectors; the index column is "
            f"{EMBEDDING_DIMENSIONS}-d (docs/ADR/0019) — a different model needs a schema migration"
        )
    return provider


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Process-wide singleton. Raises instead of falling back — readiness surfaces it."""
    return build_embedding_provider(get_settings())
