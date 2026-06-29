"""Lightweight remote change detection via HTTP HEAD (FR-2)."""

from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger


def head_metadata(url: str, *, timeout: float = 30.0, retries: int = 3) -> dict[str, Any]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; AutoSafety-RAG/1.0; +https://github.com/local/registry)"
        )
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True, verify=True) as client:
                response = client.head(url, headers=headers)
                if response.status_code == 405:
                    response = client.get(url, headers={**headers, "Range": "bytes=0-0"})
                response.raise_for_status()
                return {
                    "etag": response.headers.get("etag"),
                    "last_modified": response.headers.get("last-modified"),
                    "content_length": response.headers.get("content-length"),
                }
        except Exception as exc:
            last_err = exc
            wait = 2**attempt
            logger.warning(f"HEAD check failed for {url} (attempt {attempt + 1}): {exc}")
            time.sleep(wait)
    raise RuntimeError(f"HEAD check exhausted retries for {url}: {last_err}")


def remote_changed(
    current: dict[str, Any] | None,
    remote: dict[str, Any],
) -> bool:
    """Return True when remote metadata indicates a new revision."""
    if not current:
        return True
    etag_cur, etag_new = current.get("etag"), remote.get("etag")
    if etag_cur and etag_new:
        return etag_new.strip('"') != etag_cur.strip('"')
    lm_cur, lm_new = current.get("last_modified"), remote.get("last_modified")
    if lm_cur and lm_new:
        return lm_new != lm_cur
    cl_cur, cl_new = current.get("content_length"), remote.get("content_length")
    if cl_cur and cl_new:
        return str(cl_cur) != str(cl_new)
    return True
