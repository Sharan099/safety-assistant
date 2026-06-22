"""Answer cache for evaluation — generate once, reuse everywhere."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from config import EVALUATION_DIR

CACHE_DIR = EVALUATION_DIR / "cache"
CACHE_FILE = CACHE_DIR / "answers.json"


def load_cache(path: Path | None = None) -> dict[str, Any]:
    p = path or CACHE_FILE
    if not p.exists():
        return {"_meta": {"version": 1}, "items": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_cache(data: dict[str, Any], path: Path | None = None) -> Path:
    p = path or CACHE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def get_cached_item(cache: dict, item_id: str) -> dict | None:
    return (cache.get("items") or {}).get(item_id)


def build_or_load_answers(
    items: list[dict],
    *,
    generate_fn: Callable[[dict], dict],
    from_cache_only: bool = False,
    skip_llm: bool = False,
    delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """
    Return cache dict with answer + contexts per item id.
    generate_fn(item) -> {answer, contexts, documents, ...}
    """
    cache = load_cache()
    cache.setdefault("items", {})
    cache.setdefault("_meta", {})["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for item in items:
        iid = item["id"]
        if iid in cache["items"] and cache["items"][iid].get("answer") is not None:
            continue
        if from_cache_only or skip_llm:
            continue
        result = generate_fn(item)
        cache["items"][iid] = {
            "question": item["question"],
            "answer": result.get("answer", ""),
            "contexts": result.get("contexts", []),
            "documents": result.get("documents", []),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tokens": result.get("tokens", {}),
        }
        save_cache(cache)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return cache
