"""Load `knowledge/00_registry/source_manifest.yaml` — the only sources the
ingestion pipeline is allowed to treat as available (PRD.md §6, TRD.md §12).
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

MANIFEST_PATH = pathlib.Path(__file__).resolve().parents[2] / "knowledge" / "00_registry" / "source_manifest.yaml"


def load_manifest(path: pathlib.Path = MANIFEST_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest: dict[str, Any] = yaml.safe_load(f)
        return manifest


def get_source(source_id: str, path: pathlib.Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = load_manifest(path)
    for source in manifest["sources"]:
        if source["source_id"] == source_id:
            return dict(source)
    raise KeyError(f"unknown source_id (not in manifest): {source_id}")
