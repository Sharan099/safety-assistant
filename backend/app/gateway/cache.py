"""In-process response cache for repeat gateway queries."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    text: str
    model_key: str
    model_id: str
    provider: str
    prompt_tokens: int
    completion_tokens: int


_lock = threading.Lock()
_store: dict[str, CacheEntry] = {}


def _cache_key(messages: list[dict[str, str]], model_key: str) -> str:
    payload = model_key + "|" + "|".join(f"{m.get('role')}:{m.get('content','')}" for m in messages)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(messages: list[dict[str, str]], model_key: str) -> CacheEntry | None:
    key = _cache_key(messages, model_key)
    with _lock:
        return _store.get(key)


def put(
    messages: list[dict[str, str]],
    model_key: str,
    *,
    text: str,
    model_id: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    key = _cache_key(messages, model_key)
    with _lock:
        _store[key] = CacheEntry(
            text=text,
            model_key=model_key,
            model_id=model_id,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def clear_for_tests() -> None:
    with _lock:
        _store.clear()
