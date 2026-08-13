"""Validate the knowledge-source manifest against the files on disk.

Covers TRD.md §12/§13: every registered source must have a correct SHA-256,
and the canonical copy must be byte-identical to the original. This is the
Phase-0 registry check; the full ingestion pipeline (extraction, chunking,
embeddings) is Phase 4.
"""

import hashlib
import pathlib
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "knowledge" / "00_registry" / "source_manifest.yaml"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        loaded: dict[str, Any] = yaml.safe_load(f)
        return loaded


def test_manifest_loads() -> None:
    manifest = _load_manifest()
    assert manifest["schema_version"] == 1
    # 11 original V1 sources + 17 Level-3 real-corpus sources (PRD_LEVEL3.md
    # §4) registered by scripts/profile_knowledge_sources.py, 2026-08-13.
    assert len(manifest["sources"]) == 28


def test_every_source_has_matching_original_hash_and_size() -> None:
    # The original is the one thing every registered source must have,
    # canonical copy or not (docs/ADR/0011: large archives are not
    # duplicated into knowledge/ given disk headroom).
    manifest = _load_manifest()
    for source in manifest["sources"]:
        original = ROOT / source["original_path"]
        assert original.is_file(), f"missing original: {original}"
        assert original.stat().st_size == source["size_bytes"], source["source_id"]
        assert _sha256(original) == source["sha256"], source["source_id"]


def test_every_canonical_copy_that_exists_matches_the_original() -> None:
    manifest = _load_manifest()
    for source in manifest["sources"]:
        if source.get("canonical_path") is None:
            continue
        canonical = ROOT / source["canonical_path"]
        assert canonical.is_file(), f"missing canonical copy: {canonical}"
        assert canonical.stat().st_size == source["size_bytes"], source["source_id"]
        assert _sha256(canonical) == source["sha256"], source["source_id"]


def test_no_source_marked_ingested_yet() -> None:
    # Phase-0 truth: nothing has gone through the ingestion pipeline yet.
    manifest = _load_manifest()
    for source in manifest["sources"]:
        assert source["processing_status"] == "NOT_INGESTED", source["source_id"]
