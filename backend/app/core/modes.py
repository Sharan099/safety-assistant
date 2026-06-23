"""Load and resolve use-case mode configuration from config/modes.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_MODES_PATH = Path(__file__).resolve().parents[3] / "config" / "modes.yaml"


@dataclass(frozen=True)
class ModeConfig:
    name: str
    label: str
    doc_type_scope: tuple[str, ...]
    hard_filters: dict[str, Any]
    soft_boosts: dict[str, float]
    retrieval_k: int
    llm_tier_override: int
    prompt_template_name: str
    grounding_min_semantic: float
    grounding_min_rerank_prob: float
    allow_synthetic: bool
    temperature: float
    force_strong_reranker: bool = False


def _parse_mode(name: str, raw: dict[str, Any]) -> ModeConfig:
    return ModeConfig(
        name=name,
        label=str(raw.get("label", name)),
        doc_type_scope=tuple(raw.get("doc_type_scope", ["legal"])),
        hard_filters=dict(raw.get("hard_filters", {})),
        soft_boosts={k: float(v) for k, v in raw.get("soft_boosts", {}).items()},
        retrieval_k=int(raw.get("retrieval_k", 8)),
        llm_tier_override=int(raw.get("llm_tier_override", 2)),
        prompt_template_name=str(raw.get("prompt_template_name", "clause_citation")),
        grounding_min_semantic=float(raw.get("grounding_min_semantic", 0.35)),
        grounding_min_rerank_prob=float(raw.get("grounding_min_rerank_prob", 0.25)),
        allow_synthetic=bool(raw.get("allow_synthetic", False)),
        temperature=float(raw.get("temperature", 0)),
        force_strong_reranker=bool(raw.get("force_strong_reranker", False)),
    )


@lru_cache(maxsize=1)
def load_modes() -> dict[str, ModeConfig]:
    if not _MODES_PATH.is_file():
        return {}
    data = yaml.safe_load(_MODES_PATH.read_text(encoding="utf-8")) or {}
    modes_raw = data.get("modes", {})
    return {name: _parse_mode(name, cfg) for name, cfg in modes_raw.items()}


def get_default_mode() -> str:
    if not _MODES_PATH.is_file():
        return "regulation_lookup"
    data = yaml.safe_load(_MODES_PATH.read_text(encoding="utf-8")) or {}
    return str(data.get("default_mode", "regulation_lookup"))


def get_mode(name: str | None) -> ModeConfig:
    modes = load_modes()
    key = (name or get_default_mode()).strip()
    if key in modes:
        return modes[key]
    if modes:
        return modes[get_default_mode()]
    return ModeConfig(
        name="regulation_lookup",
        label="Regulation lookup",
        doc_type_scope=("legal",),
        hard_filters={"doc_type": ["legal"]},
        soft_boosts={},
        retrieval_k=8,
        llm_tier_override=2,
        prompt_template_name="clause_citation",
        grounding_min_semantic=0.35,
        grounding_min_rerank_prob=0.25,
        allow_synthetic=False,
        temperature=0,
    )


def list_modes() -> list[dict[str, str]]:
    return [{"id": m.name, "label": m.label} for m in load_modes().values()]
