"""Stage 3 gate: structure-aware chunking + metadata contract + expansion cap."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import DocItemLabel

from ingestion.chunk import (
    assert_parent_links_resolve,
    chunk_document,
    finalize_chunks,
)
from ingestion.enrich import enrich_chunks
from ingestion.extract import load_docling_json, remediate_clause_as_caption
from ingestion.models import Chunk
from ingestion.wipe import METADATA_CONTRACT_FIELDS
from retrieval.context_budget import approx_tokens, max_parent_expand_tokens
from retrieval.expand import expand_to_parents
from retrieval.retrieve import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]
DOCLING_DIR = ROOT / "data" / "docling"

REGULATIONS = (
    ("UN_R94", "UN-ECE-R94", "Rev.3"),
    ("UN_R95", "UN-ECE-R95", "Rev.3"),
    ("UN_R16", "UN-ECE-R16", "Rev.10"),
    ("UN_R129", "UN-ECE-R129", "Rev.4"),
)


def _chunk_reg(stem: str, regulation_id: str, revision: str) -> list[Chunk]:
    path = DOCLING_DIR / f"{stem}.docling.json"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    doc = load_docling_json(path)
    remediate_clause_as_caption(doc)
    chunks = chunk_document(doc, regulation_id=regulation_id, revision=revision)
    return enrich_chunks(chunks)


def test_finalize_rejects_empty_section_number():
    raw = [
        Chunk(
            chunk_id="a",
            text="body",
            regulation_id="UN-ECE-R94",
            revision="Rev.3",
            section_number="",
            section_title="",
            section_id="UN-ECE-R94::x",
            page_number=3,
        )
    ]
    out = finalize_chunks(raw, regulation_id="UN-ECE-R94")
    assert out[0].section_number
    assert not out[0].section_number.strip() == ""


def test_payload_includes_metadata_contract():
    ch = Chunk(
        chunk_id="c1",
        text="The FFC shall not exceed the force-time curve.",
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
        section_number="5.2.1.6",
        section_title="Femur force",
        section_id="UN-ECE-R94::5.2.1.6",
        parent_section_id="UN-ECE-R94::5.2.1",
        content_type="clause",
        page_number=12,
        bounding_box=[1.0, 2.0, 3.0, 4.0],
    )
    payload = ch.payload()
    for key in METADATA_CONTRACT_FIELDS:
        assert key in payload, key
    assert payload["element_type"] == "clause"
    assert payload["coordinates"] == [1.0, 2.0, 3.0, 4.0]
    assert payload["section"] == "Femur force"
    assert payload["parent_id"] == "UN-ECE-R94::5.2.1"
    assert payload["document_id"] == "UN-ECE-R94"


def test_enrichment_header_prepended():
    chunks = [
        Chunk(
            chunk_id="c1",
            text="Limit text.",
            regulation_id="UN-ECE-R94",
            revision="Rev.3",
            section_number="5.2.1.6",
            section_title="Femur",
            section_id="UN-ECE-R94::5.2.1.6",
        )
    ]
    enriched = enrich_chunks(chunks)
    assert enriched[0].enriched_text.startswith("From UN-ECE-R94 Rev.3 §5.2.1.6")
    assert "PERFORMANCE LIMIT" in enriched[0].enriched_text.splitlines()[0]


def test_expansion_never_exceeds_size_cap(monkeypatch):
    """Oversized parent (whole-Annex class) is rejected; leaf kept."""
    monkeypatch.setenv("MAX_PARENT_EXPAND_TOKENS", "50")
    # Force reload of cap reader
    from retrieval import context_budget

    monkeypatch.setattr(context_budget, "DEFAULT_MAX_PARENT_EXPAND_TOKENS", 50)

    leaf = RetrievedChunk(
        chunk_id="leaf1",
        text="short leaf about 5.2.1.6 femur force",
        score=0.9,
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
        section_number="5.2.1.6",
        section_title="Femur",
        section_id="UN-ECE-R94::5.2.1.6",
        parent_section_id="UN-ECE-R94::Annex 4",
        page_number=12,
        bounding_box=[],
        content_type="clause",
    )
    huge = "WORD " * 5000  # far above 50-token cap
    parent = RetrievedChunk(
        chunk_id="parent1",
        text=huge,
        score=0.5,
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
        section_number="Annex 4",
        section_title="Annex 4",
        section_id="UN-ECE-R94::Annex 4",
        parent_section_id=None,
        page_number=30,
        bounding_box=[],
        content_type="clause",
    )

    client = MagicMock()

    def _scroll(**kwargs):  # noqa: ANN003
        # Return parent when filtering by section_id == Annex
        filt = kwargs.get("scroll_filter") or kwargs.get("filter")
        # expand uses client.scroll with filter in points_selector style via helper
        return [SimpleNamespace(payload=parent.model_dump(), id="p1")], None

    # Patch the helper used inside expand_to_parents
    import retrieval.expand as expand_mod

    def fake_scroll_filter(client, *, collection, key, value):  # noqa: ANN001
        if key == "section_id" and value == "UN-ECE-R94::Annex 4":
            return [parent]
        if key == "parent_section_id":
            return []
        return []

    monkeypatch.setattr(expand_mod, "_scroll_filter", fake_scroll_filter)
    monkeypatch.setattr(expand_mod, "max_parent_expand_tokens", lambda: 50)

    out = expand_to_parents([leaf], client=client, collection="regulations")
    assert len(out) == 1
    assert out[0].chunk_id == "leaf1"
    assert approx_tokens(out[0].text) <= approx_tokens(huge)


@pytest.mark.parametrize("stem,regulation_id,revision", REGULATIONS)
def test_gate3_no_null_section_number(stem, regulation_id, revision):
    chunks = _chunk_reg(stem, regulation_id, revision)
    assert chunks, f"{stem}: no chunks"
    empty = [c.chunk_id for c in chunks if not (c.section_number or "").strip()]
    assert empty == [], f"{stem}: empty section_number on {empty[:5]}"


@pytest.mark.parametrize("stem,regulation_id,revision", REGULATIONS)
def test_gate3_tables_atomic(stem, regulation_id, revision):
    chunks = _chunk_reg(stem, regulation_id, revision)
    tables = [c for c in chunks if c.content_type == "table"]
    if not tables:
        pytest.skip(f"{stem}: no table chunks")
    for t in tables:
        # Atomic = one chunk owns the full markdown table (has header separator).
        assert "|" in t.text
        # Must not look like a mid-row orphan (single pipe line without structure).
        lines = [ln for ln in t.text.splitlines() if ln.strip().startswith("|")]
        assert len(lines) >= 2, t.section_id


@pytest.mark.parametrize("stem,regulation_id,revision", REGULATIONS)
def test_gate3_parent_id_resolves(stem, regulation_id, revision):
    chunks = _chunk_reg(stem, regulation_id, revision)
    assert_parent_links_resolve(chunks)


@pytest.mark.parametrize("stem,regulation_id,revision", REGULATIONS)
def test_gate3_metadata_contract_on_payload(stem, regulation_id, revision):
    chunks = _chunk_reg(stem, regulation_id, revision)
    sample = chunks[:50]
    for ch in sample:
        payload = ch.payload()
        for key in METADATA_CONTRACT_FIELDS:
            assert key in payload
        assert payload["section_number"]
        assert payload["element_type"]
        assert payload["document_id"]
        assert payload["element_id"]
