"""Fallback message must only name regulations present in the live catalog."""

from __future__ import annotations

from generation.answer import not_found_in_regulations_message
from retrieval.retrieve import IndexedRegulation


def test_fallback_never_mentions_absent_regulation():
    live = [
        IndexedRegulation(regulation_id="UN-ECE-R94", revision="Rev.3", chunk_count=113),
    ]
    msg = not_found_in_regulations_message(indexed=live)

    assert "R94" in msg
    assert "couldn't find relevant content" in msg.lower()
    assert "clause number" in msg.lower()
    assert "upload the relevant regulation" in msg.lower()
    for absent in ("R95", "R16", "R129", "FMVSS"):
        assert absent not in msg, f"fallback unexpectedly mentions {absent}: {msg}"


def test_fallback_empty_index():
    msg = not_found_in_regulations_message(indexed=[])
    assert "none indexed" in msg.lower()
    assert "R94" not in msg
    assert "R95" not in msg
