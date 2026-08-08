"""Regression: small-to-big must keep numeric child clauses under a parent shell."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from retrieval.expand import expand_to_parents
from retrieval.retrieve import RetrievedChunk


def _chunk(**kwargs) -> RetrievedChunk:
    defaults = dict(
        chunk_id="x",
        text="text",
        regulation_id="UN-ECE-R94",
        section_number="5.2.8.1.4.1",
        section_id="UN-ECE-R94::5.2.8.1.4.1",
        parent_section_id="UN-ECE-R94::5.2.8.1.4",
        score=1.0,
    )
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


def test_expand_merges_numeric_children_under_parent_shell():
    """Parent shell alone lacks '100'; children under parent_section_id must merge in."""
    leaf = _chunk(
        chunk_id="leaf_4_1",
        text="5.2.8.1.4.1 Electrical isolation shall be at least 100 ohm.",
        section_number="5.2.8.1.4.1",
        section_id="UN-ECE-R94::5.2.8.1.4.1",
        parent_section_id="UN-ECE-R94::5.2.8.1.4",
    )
    shell = _chunk(
        chunk_id="shell",
        text="5.2.8.1.4 Isolation resistance\n\nThe criteria specified below shall be met.",
        section_number="5.2.8.1.4",
        section_id="UN-ECE-R94::5.2.8.1.4",
        parent_section_id="UN-ECE-R94::5.2.8.1",
    )
    sibling = _chunk(
        chunk_id="leaf_4_2",
        text="5.2.8.1.4.2 Isolation resistance shall be at least 500 ohm for certain buses.",
        section_number="5.2.8.1.4.2",
        section_id="UN-ECE-R94::5.2.8.1.4.2",
        parent_section_id="UN-ECE-R94::5.2.8.1.4",
    )

    client = MagicMock()

    def _scroll(**kwargs):
        filt = kwargs.get("scroll_filter")
        # qdrant Filter.must[0].match.value
        cond = filt.must[0]
        key = cond.key
        value = cond.match.value
        if key == "section_id" and value == "UN-ECE-R94::5.2.8.1.4":
            return [SimpleNamespace(payload=shell.model_dump()),], None
        if key == "parent_section_id" and value == "UN-ECE-R94::5.2.8.1.4":
            return [
                SimpleNamespace(payload=leaf.model_dump()),
                SimpleNamespace(payload=sibling.model_dump()),
            ], None
        return [], None

    client.scroll.side_effect = _scroll

    out = expand_to_parents([leaf], client=client, collection="regulations")
    assert len(out) == 1
    merged = out[0]
    assert "100" in (merged.text or "")
    assert "500" in (merged.text or "")
    assert "Isolation resistance" in (merged.text or "")
