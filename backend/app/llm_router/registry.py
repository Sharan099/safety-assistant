"""Groq model registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from backend.app.llm_router.config import RouterConfig, api_keys


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str  # groq
    model_id: str
    priority: int
    max_context_tokens: int
    effective_request_tokens: int

    def is_available(self) -> bool:
        return bool(api_keys().get(self.provider))


def _build_specs() -> dict[str, ModelSpec]:
    # Free Groq models only. Primary generation default: openai/gpt-oss-120b.
    groq_primary = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_qwen = os.getenv("GROQ_MODEL_QWEN", "qwen/qwen3-32b")
    groq_instant = os.getenv("GROQ_MODEL_INSTANT", "llama-3.1-8b-instant")

    specs = [
        ModelSpec("groq_70b", "groq", groq_primary, 1, 128_000, 12_000),
        ModelSpec("groq_qwen3", "groq", groq_qwen, 2, 128_000, 12_000),
        ModelSpec("groq_instant", "groq", groq_instant, 3, 128_000, 6_000),
    ]
    reg = {s.key: s for s in specs}
    reg["groq"] = reg["groq_70b"]
    reg["groq_fast"] = reg["groq_70b"]
    return reg


REGISTRY: dict[str, ModelSpec] = _build_specs()


def get_spec(key: str) -> ModelSpec | None:
    return REGISTRY.get(key)


def ordered_chain(config: RouterConfig | None = None) -> list[ModelSpec]:
    cfg = config or RouterConfig.from_env()
    out: list[ModelSpec] = []
    seen: set[str] = set()
    for key in cfg.chain:
        spec = get_spec(key)
        if spec is None or spec.key in seen:
            continue
        seen.add(spec.key)
        out.append(spec)
    return out


@lru_cache(maxsize=1)
def filter_groq_available(probe: bool) -> frozenset[str]:
    if not probe or not api_keys().get("groq"):
        return frozenset()
    try:
        from backend.app.llm_router.providers.groq_provider import GroqProvider

        return frozenset(GroqProvider().list_models())
    except Exception:
        return frozenset()
