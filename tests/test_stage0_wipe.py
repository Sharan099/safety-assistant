"""Stage 0 gate: Qdrant wiped empty + known cache stores report empty."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stage0_report():
    """Run the wipe once for this module (idempotent)."""
    # Avoid docker restart flapping during pytest collection/reruns if env set.
    restart = os.getenv("STAGE0_RESTART_PORTKEY", "0") not in {"0", "false", "False"}
    from ingestion.wipe import run_stage0_wipe

    return run_stage0_wipe(restart_portkey=restart)


def test_qdrant_collection_exists_and_empty(stage0_report):
    from ingestion.embed_upsert import DEFAULT_COLLECTION, get_qdrant_client

    q = stage0_report["qdrant"]
    assert q["points_count"] == 0
    assert q["scroll_count"] == 0

    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    client = get_qdrant_client()
    try:
        names = {c.name for c in client.get_collections().collections}
        assert collection in names
        info = client.get_collection(collection)
        assert int(info.points_count or 0) == 0
        points, _ = client.scroll(
            collection_name=collection,
            limit=10,
            with_payload=False,
            with_vectors=False,
        )
        assert points == []
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def test_metadata_contract_fields_listed(stage0_report):
    from ingestion.wipe import METADATA_CONTRACT_FIELDS

    fields = stage0_report["qdrant"]["metadata_fields"]
    for required in METADATA_CONTRACT_FIELDS:
        assert required in fields


def test_response_cache_empty(stage0_report):
    rc = stage0_report["response_cache"]
    assert rc["empty"] is True
    assert rc["remaining"] == 0
    from cache.response_cache import count_entries

    assert count_entries() == 0


def test_llm_disk_cache_empty(stage0_report):
    llm = stage0_report["llm_disk_cache"]
    assert llm["empty"] is True
    assert llm["remaining"] == 0
    cache_dir = Path(llm["path"])
    if cache_dir.is_dir():
        assert not any(p.is_file() for p in cache_dir.rglob("*"))


def test_answer_cache_dir_empty(stage0_report):
    ac = stage0_report["answer_cache_dir"]
    assert ac["empty"] is True
    assert ac["remaining"] == 0


def test_cache_version_reset(stage0_report):
    assert stage0_report["cache_version"]["version"] == 0
    path = ROOT / "data" / "cache_version.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert int(data["version"]) == 0


def test_inprocess_caches_cleared(stage0_report):
    status = stage0_report["inprocess"]
    for key in (
        "gateway_cache",
        "prompt_cache",
        "retrieval_cache",
        "indexed_regulations_cache",
    ):
        assert key in status
        assert status[key] == "cleared", status


def test_eval_results_archived(stage0_report):
    arch = stage0_report["eval_archive"]
    assert arch["empty"] is True
    dest = Path(arch["dest"])
    assert dest.is_dir()
    # Prior run folder (or rename) should live under the archive.
    assert any(dest.iterdir()), "archive dir should contain moved artifacts"
    src = Path(arch["src"])
    leftovers = [p.name for p in src.iterdir() if p.name != "RUNS_STATUS.md"]
    assert leftovers == []
