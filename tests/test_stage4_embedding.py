"""Stage 4 gate: Qdrant point counts, hybrid vectors, tibia + figure spot-checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ingestion.embed_upsert import (
    DEFAULT_COLLECTION,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    get_qdrant_client,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data" / "stage4_rebuild_report.json"

REGULATIONS = ("UN-ECE-R94", "UN-ECE-R95", "UN-ECE-R16", "UN-ECE-R129")


@pytest.fixture(scope="module")
def rebuild_report():
    """Load Stage 4 report; run rebuild once if missing / empty collection."""
    client = get_qdrant_client()
    try:
        coll = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
        names = {c.name for c in client.get_collections().collections}
        points = 0
        if coll in names:
            points = int(client.get_collection(coll).points_count or 0)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    if points == 0 or not REPORT_PATH.is_file():
        from ingestion.stage4_rebuild import rebuild_all

        report = rebuild_all(skip_upsert=False)
    else:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return report


def test_stage4_report_has_four_regs(rebuild_report):
    expected = rebuild_report["expected_chunk_counts"]
    for rid in REGULATIONS:
        assert rid in expected
        assert expected[rid] > 0


def test_qdrant_point_count_matches_expected(rebuild_report):
    client = get_qdrant_client()
    coll = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    try:
        info = client.get_collection(coll)
        total = int(info.points_count or 0)
        assert total == int(rebuild_report["total_chunks"])

        from qdrant_client.http import models as qm

        for rid, expected in rebuild_report["expected_chunk_counts"].items():
            counted = client.count(
                collection_name=coll,
                count_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="regulation_id",
                            match=qm.MatchValue(value=rid),
                        )
                    ]
                ),
                exact=True,
            )
            assert int(counted.count) == int(expected), (rid, counted.count, expected)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def test_dense_and_sparse_vectors_present(rebuild_report):
    client = get_qdrant_client()
    coll = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    try:
        pts, _ = client.scroll(
            collection_name=coll,
            limit=5,
            with_vectors=True,
            with_payload=True,
        )
        assert pts, "collection empty"
        for pt in pts:
            vectors = pt.vector
            assert isinstance(vectors, dict), vectors
            assert DENSE_VECTOR in vectors
            assert SPARSE_VECTOR in vectors
            dense = vectors[DENSE_VECTOR]
            assert dense is not None and len(dense) > 0
            sparse = vectors[SPARSE_VECTOR]
            # SparseVector has indices/values
            indices = getattr(sparse, "indices", None)
            values = getattr(sparse, "values", None)
            if indices is None and isinstance(sparse, dict):
                indices = sparse.get("indices")
                values = sparse.get("values")
            assert indices is not None and len(indices) > 0
            assert values is not None and len(values) > 0
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def test_tibia_force_spot_check_section(rebuild_report):
    """Retrieval spot-check: tibia/TCFC lands on corrected clause, not a caption."""
    from retrieval.retrieve import hybrid_search

    hits = hybrid_search(
        "UN R94 tibia compression force criterion TCFC limit 8 kN",
        regulation_id="UN-ECE-R94",
        top_k=10,
    )
    assert hits, "no hits for tibia-force query"
    top = hits[0]
    # Must not be an empty/page-fallback section; prefer 5.2.1.7 when present.
    assert top.section_number
    assert top.page_number is not None
    joined = " ".join(f"{h.section_number}:{h.text[:80]}" for h in hits[:5])
    assert "5.2.1.7" in joined or "TCFC" in joined.upper() or "tibia" in joined.lower()
    # Corrected attribution: top hit should not claim figure-caption content_type.
    assert top.content_type != "caption"


def test_figure_dependent_query_returns_figure_chunk(rebuild_report):
    from retrieval.retrieve import hybrid_search

    hits = hybrid_search(
        "femur force force-time performance curve Figure 3",
        regulation_id="UN-ECE-R94",
        top_k=15,
    )
    assert hits
    figure_hits = [h for h in hits if h.content_type == "figure"]
    assert figure_hits, (
        "expected a Stage-2 figure chunk for femur force curve; "
        f"got content_types={[h.content_type for h in hits[:8]]}"
    )
    assert any(
        "5.2.1.6" in (h.section_number or "") or "femur" in (h.text or "").lower()
        for h in figure_hits
    )
