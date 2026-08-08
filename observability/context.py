"""ContextVar so retrieve/embed/rerank can attach metrics without signature churn."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from observability.trace import QueryTrace

_current: ContextVar[QueryTrace | None] = ContextVar("query_trace", default=None)


def set_current_trace(trace: QueryTrace | None):
    return _current.set(trace)


def reset_current_trace(token) -> None:
    _current.reset(token)


def get_current_trace() -> QueryTrace | None:
    return _current.get()
