"""Static system-prompt caching: reuse prefix tokens across answer calls."""

from __future__ import annotations

import hashlib
import os
import threading

_lock = threading.Lock()
_seen: dict[str, int] = {}  # prompt_hash -> estimated tokens


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token heuristic when usage is unavailable
    return max(1, len(text) // 4)


def prompt_cache_lookup(system_prompt: str) -> tuple[bool, int]:
    """Return (hit, tokens_saved). First call warms the cache; later hits save tokens."""
    if os.getenv("PROMPT_CACHE", "1") in {"0", "false", "False"}:
        return False, 0
    key = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    tokens = _estimate_tokens(system_prompt)
    with _lock:
        if key in _seen:
            return True, _seen[key]
        _seen[key] = tokens
        return False, 0


def reset_prompt_cache() -> None:
    with _lock:
        _seen.clear()
