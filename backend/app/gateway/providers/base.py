"""Provider error types for gateway routing."""

from __future__ import annotations

from typing import Literal


ErrorKindName = Literal[
    "too_large", "rate_limit", "timeout", "connection", "decommissioned", "fatal"
]


class ProviderError(Exception):
    def __init__(self, message: str, *, kind: ErrorKindName | None = None, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
