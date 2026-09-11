from __future__ import annotations

from functools import lru_cache

from safety_assistant.config import Settings, get_settings
from safety_assistant.providers.rerankers.base import Reranker


def build_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker == "none":
        return None
    if settings.reranker == "cross_encoder":
        from safety_assistant.providers.rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker()
    from safety_assistant.providers.rerankers.heuristic import LexicalAuthorityReranker

    return LexicalAuthorityReranker()


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    return build_reranker(get_settings())
