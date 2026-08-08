"""Process-wide cache version — bumps on every successful regulation ingest.

Included in Portkey ``x-portkey-cache-namespace`` (rewrite/answer) and in the
optional SQLite ``response_cache`` key space so newly ingested PDFs never reuse
stale cached completions or pipeline answers.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = ROOT / "data" / "cache_version.json"

_lock = threading.Lock()


def get_cache_version() -> int:
    """Current ingest generation (included in answer-cache keys)."""
    with _lock:
        if not VERSION_PATH.is_file():
            return 0
        try:
            data = json.loads(VERSION_PATH.read_text(encoding="utf-8"))
            return int(data.get("version") or 0)
        except Exception:  # noqa: BLE001
            return 0


def bump_cache_version() -> int:
    """Increment after a successful upsert; returns the new version."""
    with _lock:
        VERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        current = 0
        if VERSION_PATH.is_file():
            try:
                current = int(json.loads(VERSION_PATH.read_text(encoding="utf-8")).get("version") or 0)
            except Exception:  # noqa: BLE001
                current = 0
        nxt = current + 1
        VERSION_PATH.write_text(
            json.dumps({"version": nxt}, indent=2),
            encoding="utf-8",
        )
        return nxt
