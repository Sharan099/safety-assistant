"""Shared httpx timeouts — short connect, longer read."""

from __future__ import annotations

import os

import httpx

CONNECT_TIMEOUT_S = float(os.getenv("GATEWAY_CONNECT_TIMEOUT", "3"))
READ_TIMEOUT_S = float(os.getenv("GATEWAY_READ_TIMEOUT", "60"))


def provider_timeout(*, read: float | None = None) -> httpx.Timeout:
    """Fail fast on TCP connect; allow full read window for generation."""
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_S,
        read=read or READ_TIMEOUT_S,
        write=30.0,
        pool=CONNECT_TIMEOUT_S,
    )
