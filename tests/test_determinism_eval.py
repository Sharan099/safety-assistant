"""Determinism: identical retrieval across repeated asks."""

from __future__ import annotations

from eval.determinism_eval import _chunk_fingerprint


def test_chunk_fingerprint_ordered():
    class C:
        def __init__(self, chunk_id: str) -> None:
            self.chunk_id = chunk_id

    assert _chunk_fingerprint([C("a"), C("b")]) == ("a", "b")
    assert _chunk_fingerprint([C("a"), C("")]) == ("a",)
