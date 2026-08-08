"""Exact + semantic response cache and cache_version invalidation."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.cache_version import bump_cache_version, get_cache_version
from cache.response_cache import (
    cache_key,
    cosine_similarity,
    get_exact,
    get_semantic,
    init_db,
    lookup,
    put_exact,
    store,
)


@pytest.fixture()
def cache_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "response_cache.sqlite3"
    ver = tmp_path / "cache_version.json"
    monkeypatch.setenv("RESPONSE_CACHE_DB", str(db))
    monkeypatch.setenv("ANSWER_CACHE", "1")
    monkeypatch.setenv("SEMANTIC_CACHE", "0")  # exact tests: no embedder
    monkeypatch.setattr("api.cache_version.VERSION_PATH", ver)
    import cache.response_cache as rc

    rc._db_path = None
    init_db()
    yield tmp_path
    rc._db_path = None


def test_exact_hit_and_miss(cache_env: Path):
    q = "Which vehicles are covered under UN R95?"
    payload = {
        "answer": "I could not find this in the indexed regulations (UN-ECE-R94).",
        "sources": [],
        "model": "mock",
        "provider": "mock",
        "not_found": True,
        "input_tokens": 120,
        "output_tokens": 40,
        "cost_usd": 0.002,
    }
    put_exact(q, payload, regulation_id="UN-ECE-R95")
    hit = get_exact(q, regulation_id="UN-ECE-R95")
    assert hit is not None
    assert hit["not_found"] is True
    assert "could not find" in hit["answer"].lower()
    assert hit["cache_hit"] == "exact"
    # Different regulation filter → miss
    assert get_exact(q, regulation_id="UN-ECE-R94") is None


def test_ingest_bump_invalidates_stale_not_found(cache_env: Path):
    """Cache a R95 not-found, fake-ingest (bump version), assert stale miss."""
    q = "Which vehicles are covered under UN R95?"
    put_exact(
        q,
        {
            "answer": "I could not find this in the indexed regulations (UN-ECE-R94).",
            "sources": [],
            "model": "mock",
            "provider": "mock",
            "not_found": True,
        },
        regulation_id="UN-ECE-R95",
    )
    assert get_exact(q, regulation_id="UN-ECE-R95") is not None
    k0 = cache_key(q, regulation_id="UN-ECE-R95")
    v0 = get_cache_version()

    # Fake successful R95 ingestion.
    bump_cache_version()
    assert get_cache_version() == v0 + 1
    k1 = cache_key(q, regulation_id="UN-ECE-R95")
    assert k0 != k1
    assert get_exact(q, regulation_id="UN-ECE-R95") is None


def test_served_from_cache_zeros_cost(cache_env: Path, monkeypatch: pytest.MonkeyPatch):
    from generation.answer import answer_question
    from generation.llm_client import LLMClient

    q = "What is the ThCC limit in R94?"
    put_exact(
        q,
        {
            "answer": "42 mm [R94 §5.2.1.4, p.12]",
            "sources": [
                {
                    "chunk_id": "c1",
                    "regulation_id": "UN-ECE-R94",
                    "section_number": "5.2.1.4",
                    "section_title": "ThCC",
                    "page_number": 12,
                    "bounding_box": [],
                    "content_type": "clause",
                    "text": "ThCC shall not exceed 42 mm",
                    "score": 0.9,
                    "citation": "[R94 §5.2.1.4, p.12]",
                }
            ],
            "model": "llama-cached",
            "provider": "groq",
            "not_found": False,
            "input_tokens": 900,
            "output_tokens": 80,
            "cost_usd": 0.01,
        },
        regulation_id=None,
    )
    monkeypatch.setenv("CONVERSATIONS_DB", str(cache_env / "conv.sqlite3"))
    import api.conversations as conv

    conv._db_path = None

    resp = answer_question(
        q,
        llm=LLMClient(provider="mock"),
        skip_answer_cache=False,
        persist_turn=False,
        chunks=[],  # unused on cache hit
    )
    assert resp.served_from_cache is True
    assert resp.answer_cached is True
    assert resp.cache_hit_kind == "exact"
    assert resp.metrics.get("cost_usd") == 0.0
    assert resp.metrics.get("input_tokens") == 0
    assert resp.metrics.get("output_tokens") == 0
    assert "42 mm" in resp.answer


def test_semantic_hit_with_fake_embeddings(cache_env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEMANTIC_CACHE", "1")
    monkeypatch.setenv("SEMANTIC_CACHE_THRESHOLD", "0.95")

    base = [1.0, 0.0, 0.0]
    near = [0.99, 0.1, 0.0]  # high cosine with base
    far = [0.0, 1.0, 0.0]

    def fake_embed(texts):
        out = []
        for t in texts:
            low = t.lower()
            if "thcc" in low or "thorax compression" in low:
                out.append(list(near) if "maximum" in low or "limit" in low else list(base))
            else:
                out.append(list(far))
        return out

    put_exact(
        "What is the ThCC limit in UN R94?",
        {
            "answer": "42 mm",
            "sources": [],
            "model": "mock",
            "provider": "mock",
            "not_found": False,
        },
        regulation_id="UN-ECE-R94",
        embedding=base,
    )

    hit = get_semantic(
        "What is the maximum Thorax Compression Criterion limit in UN R94?",
        regulation_id="UN-ECE-R94",
        embed_fn=fake_embed,
        threshold=0.95,
    )
    assert hit is not None
    assert hit["answer"] == "42 mm"
    assert hit["cache_hit"] == "semantic"
    assert hit["semantic_similarity"] >= 0.95
    assert "ThCC" in hit["semantic_cached_question"]

    miss = get_semantic(
        "What is the i-Size definition under R129?",
        regulation_id="UN-ECE-R94",
        embed_fn=fake_embed,
        threshold=0.95,
    )
    assert miss is None


def test_semantic_disabled_flag(cache_env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEMANTIC_CACHE", "0")
    put_exact(
        "What is the ThCC limit?",
        {"answer": "42 mm", "sources": [], "not_found": False},
        regulation_id="",
        embedding=[1.0, 0.0],
    )
    assert (
        get_semantic(
            "What is the thorax compression limit?",
            regulation_id="",
            embed_fn=lambda texts: [[0.99, 0.1] for _ in texts],
            threshold=0.5,
        )
        is None
    )


def test_cosine_similarity_unit():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_lookup_prefers_exact_over_semantic(cache_env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEMANTIC_CACHE", "1")
    q = "Exact question about VC"
    put_exact(
        q,
        {"answer": "exact-answer", "sources": [], "not_found": False},
        regulation_id="",
        embedding=[1.0, 0.0],
    )
    hit = lookup(
        q,
        regulation_id="",
        embed_fn=lambda texts: [[1.0, 0.0] for _ in texts],
        allow_semantic=True,
    )
    assert hit is not None
    assert hit["cache_hit"] == "exact"
    assert hit["answer"] == "exact-answer"


def test_store_without_semantic_skips_embedder(cache_env: Path):
    store(
        "Plain store question",
        {"answer": "ok", "sources": [], "not_found": False},
        regulation_id="UN-ECE-R94",
    )
    assert get_exact("Plain store question", regulation_id="UN-ECE-R94") is not None
