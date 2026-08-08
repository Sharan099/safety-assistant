"""Eval-infra-only retries with exponential backoff (not production traffic).

Production answer/rewrite/judge paths keep ``LLMClient``'s own retry policy.
RAGAS / DeepEval / DeepTeam judge calls use this wrapper so scoring is resilient
to transient gateway blips without widening production fallback behavior.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def eval_judge_max_retries() -> int:
    raw = (os.getenv("EVAL_JUDGE_MAX_RETRIES") or "4").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 4


def eval_judge_retry_base_sec() -> float:
    raw = (os.getenv("EVAL_JUDGE_RETRY_BASE_SEC") or "1.5").strip()
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 1.5


def call_with_eval_retry(
    fn: Callable[[], T],
    *,
    label: str = "eval_judge",
    max_retries: int | None = None,
    base_sec: float | None = None,
) -> T:
    """Run ``fn`` with exponential backoff; raises the last error after exhaustion."""
    retries = eval_judge_max_retries() if max_retries is None else max(0, max_retries)
    base = eval_judge_retry_base_sec() if base_sec is None else max(0.1, base_sec)
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            sleep_s = base * (2**attempt)
            logger.warning(
                "%s failed (attempt %d/%d): %s; backoff %.1fs",
                label,
                attempt + 1,
                retries + 1,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def judge_with_eval_retry(
    client: Any,
    *,
    messages: Sequence[dict[str, str]],
    question: str = "",
    **kwargs: Any,
) -> Any:
    """``client.judge(...)`` with eval-only backoff (separate from production)."""
    return call_with_eval_retry(
        lambda: client.judge(messages=messages, question=question, **kwargs),
        label="eval_judge.judge",
    )
