"""In-process OpenRouter failover usage counter (paid tier visibility)."""

from __future__ import annotations

import threading
from datetime import date

_lock = threading.Lock()
_day: date | None = None
_request_count = 0
_prompt_tokens = 0
_completion_tokens = 0


def note_openrouter_request(*, prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    global _day, _request_count, _prompt_tokens, _completion_tokens
    today = date.today()
    with _lock:
        if _day != today:
            _day = today
            _request_count = 0
            _prompt_tokens = 0
            _completion_tokens = 0
        _request_count += 1
        _prompt_tokens += prompt_tokens
        _completion_tokens += completion_tokens
        return {
            "daily_requests": _request_count,
            "daily_prompt_tokens": _prompt_tokens,
            "daily_completion_tokens": _completion_tokens,
        }


def reset_for_tests() -> None:
    global _day, _request_count, _prompt_tokens, _completion_tokens
    with _lock:
        _day = None
        _request_count = 0
        _prompt_tokens = 0
        _completion_tokens = 0
