"""Semantic cache tests using fakeredis (exercises the NumPy scan fallback)."""

import numpy as np
import pytest

from backend.app.gateway.cache import SemanticCache

fakeredis = pytest.importorskip("fakeredis")

_VOCAB: dict[str, int] = {}


def _embed(text: str) -> np.ndarray:
    """Deterministic bag-of-words embedding so identical text -> identical vec."""
    dim = 64
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        idx = _VOCAB.setdefault(tok, len(_VOCAB))
        vec[idx % dim] += 1.0
    return vec


def _cache() -> SemanticCache:
    client = fakeredis.FakeStrictRedis()
    c = SemanticCache(embed_fn=_embed, redis_client=client)
    c.connect()
    return c


def test_store_then_hit_identical_prompt():
    c = _cache()
    c.store(prompt="UN R14 seat belt anchorage strength", answer="42 daN",
            model="groq", scope=["UN_R14"])
    hit = c.lookup("UN R14 seat belt anchorage strength", scope=["UN_R14"])
    assert hit is not None
    assert hit["answer"] == "42 daN"
    assert hit["similarity"] >= 0.95


def test_scope_isolation():
    c = _cache()
    c.store(prompt="anchorage strength requirements", answer="R14 answer",
            model="groq", scope=["UN_R14"])
    # Same prompt but different regulation scope must NOT hit.
    miss = c.lookup("anchorage strength requirements", scope=["UN_R16"])
    assert miss is None


def test_dissimilar_prompt_misses():
    c = _cache()
    c.store(prompt="seat belt anchorage test load", answer="x", model="groq")
    miss = c.lookup("what is the best pizza topping", scope=None)
    assert miss is None
