"""Load updatable token/call prices from config/prices.json."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICES = ROOT / "config" / "prices.json"

# Logical names used in Portkey configs / telemetry → prices.json keys.
_PROVIDER_ALIASES = {
    "nim": "nvidia_nim",
    "nvidia": "nvidia_nim",
    "openai": "openai",  # only if not remapped to nvidia_nim
    "gemini": "google",
    "portkey": "groq",  # legacy; prefer served_provider
}


@lru_cache(maxsize=4)
def load_prices(path: str | None = None) -> dict[str, Any]:
    prices_path = Path(path or os.getenv("PRICES_PATH") or DEFAULT_PRICES)
    if not prices_path.is_file():
        logger.warning("Prices file missing at %s — costs will be 0", prices_path)
        return {"models": {}, "providers": {}, "embeddings": {}, "rerank": {}, "unit": "per_1m_tokens"}
    data = json.loads(prices_path.read_text(encoding="utf-8"))
    return data


def clear_prices_cache() -> None:
    load_prices.cache_clear()


def _normalize_provider(provider: str | None) -> str:
    raw = (provider or "").strip().lower()
    if not raw:
        return ""
    return _PROVIDER_ALIASES.get(raw, raw)


def _lookup_model_row(
    models_map: dict[str, Any],
    model: str,
) -> dict[str, Any] | None:
    if not model:
        return None
    if model in models_map and isinstance(models_map[model], dict) and "input" in models_map[model]:
        return models_map[model]
    # basename fallback (e.g. strip org prefix)
    base = model.split("/")[-1]
    if base in models_map and isinstance(models_map[base], dict):
        return models_map[base]
    return None


def resolve_llm_price_row(
    *,
    model: str,
    provider: str | None = None,
) -> dict[str, Any]:
    """Return ``{input, output}`` USD-per-1M row for provider+model."""
    prices = load_prices()
    prov = _normalize_provider(provider)
    if prov:
        providers = prices.get("providers") or {}
        bucket = providers.get(prov)
        if isinstance(bucket, dict):
            row = _lookup_model_row(bucket, model)
            if row is not None:
                return row
            if "input" in bucket:  # single flat rate for provider
                return bucket
            default = bucket.get("default")
            if isinstance(default, dict):
                return default
    models = prices.get("models") or {}
    row = _lookup_model_row(models, model)
    if row is not None:
        return row
    return models.get("default_llm") or {"input": 0.0, "output": 0.0}


def llm_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str | None = None,
) -> float:
    """Cost from ``config/prices.json`` — prefer per-provider rows when ``provider`` is set."""
    row = resolve_llm_price_row(model=model, provider=provider)
    return (input_tokens / 1_000_000.0) * float(row.get("input") or 0.0) + (
        output_tokens / 1_000_000.0
    ) * float(row.get("output") or 0.0)


def embedding_cost_usd(*, model: str, tokens: int) -> float:
    prices = load_prices()
    emb = prices.get("embeddings") or {}
    row = emb.get(model) or emb.get("default_embedding") or {"input": 0.0}
    return (tokens / 1_000_000.0) * float(row.get("input") or 0.0)


def rerank_cost_usd(*, model: str, calls: int) -> float:
    prices = load_prices()
    rr = prices.get("rerank") or {}
    row = rr.get(model) or rr.get("default_rerank") or {"per_call": 0.0}
    return calls * float(row.get("per_call") or 0.0)
