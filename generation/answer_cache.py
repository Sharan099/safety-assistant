"""Exact-question answer cache — delegates to ``cache.response_cache``."""

from __future__ import annotations

from typing import Any

from cache.response_cache import (
    cache_key,
    get_exact,
    normalize_question,
    put_exact,
    store,
)


def get_cached_answer(question: str, *, regulation_id: str | None = None) -> dict[str, Any] | None:
    return get_exact(question, regulation_id=regulation_id)


def put_cached_answer(
    question: str,
    payload: dict[str, Any],
    *,
    regulation_id: str | None = None,
) -> None:
    # Exact write without embedding (callers that want semantic use ``store``).
    put_exact(question, payload, regulation_id=regulation_id)


__all__ = [
    "cache_key",
    "get_cached_answer",
    "normalize_question",
    "put_cached_answer",
    "store",
]
