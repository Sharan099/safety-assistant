"""In-process token usage counters (optional observability)."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class TokenTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0


_lock = threading.Lock()
_by_provider: dict[str, TokenTotals] = {}
_totals = TokenTotals()


def record(provider: str, *, input_tokens: int, output_tokens: int) -> None:
    global _totals
    with _lock:
        _totals.input_tokens += input_tokens
        _totals.output_tokens += output_tokens
        _totals.requests += 1
        bucket = _by_provider.setdefault(provider, TokenTotals())
        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        bucket.requests += 1


def snapshot() -> dict:
    with _lock:
        return {
            "total": TokenTotals(**vars(_totals)),
            "by_provider": dict(_by_provider),
        }


def reset_for_tests() -> None:
    global _totals
    with _lock:
        _totals = TokenTotals()
        _by_provider.clear()
