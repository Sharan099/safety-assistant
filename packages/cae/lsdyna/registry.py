"""Loads `keyword_registry.yaml` — the single source of truth for which
keyword roots this codebase recognizes, shared by the lightweight profiler
scan (`packages/cae/keyword_scan.py`) and the real parser
(`packages/cae/lsdyna/parser.py`) so the two never silently disagree about
what a "known root" is.
"""

from __future__ import annotations

import functools
import pathlib
from typing import Any

import yaml

REGISTRY_PATH = pathlib.Path(__file__).resolve().parent / "keyword_registry.yaml"


@functools.lru_cache(maxsize=1)
def load_registry(path: pathlib.Path = REGISTRY_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        registry: dict[str, Any] = yaml.safe_load(f)
        return registry


def registry_roots(path: pathlib.Path = REGISTRY_PATH) -> list[str]:
    return list(load_registry(path)["roots"].keys())


def structured_roots(path: pathlib.Path = REGISTRY_PATH) -> set[str]:
    roots = load_registry(path)["roots"]
    return {name for name, cfg in roots.items() if cfg.get("structured")}


def root_for(keyword: str, roots: list[str] | None = None) -> str | None:
    """A keyword "belongs to" a root if it equals the root exactly (`PART`)
    or is an underscore-suffixed variant (`PART_COMPOSITE`)."""
    candidates = roots if roots is not None else registry_roots()
    for root in candidates:
        if keyword == root or keyword.startswith(root + "_"):
            return root
    return None
