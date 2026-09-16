"""Fault injection (ENGINEERING.md §16): the system must fail safely, not silently.

- reranker crash   → retrieval still answers from fused order, marked degraded
- embedding crash  → readiness 503, /search 500 with request id (no fake vectors)
- database down    → readiness 503 with the dependency named
- event loop       → CPU-bound /search runs in the threadpool, loop stays responsive
"""

from __future__ import annotations

import asyncio
import pathlib
import time

import pytest
from fastapi.testclient import TestClient

from safety_assistant.ingestion.fetch import FilesystemBlobStore
from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.providers.embeddings.hashing import HashingEmbeddingProvider
from safety_assistant.providers.rerankers import RerankCandidate
from safety_assistant.retrieval import RetrievalService
from safety_assistant.retrieval.service import RetrievalConfig
from tests.conftest import requires_db
from tests.support.minireg import registry_for

pytestmark = requires_db


class ExplodingReranker:
    model_name = "exploding"
    model_version = "v0"

    def score(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        raise RuntimeError("model file missing")


class ExplodingEmbedder:
    model_name = "exploding"
    model_version = "v0"
    dimensions = 384

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("ONNX runtime crashed")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("ONNX runtime crashed")


@pytest.fixture
def corpus(clean_db, db_session, tmp_path: pathlib.Path):  # type: ignore[no-untyped-def]
    reg = registry_for(tmp_path, revisions=(2,))
    emb = HashingEmbeddingProvider(384)
    out = ingest_source(
        db_session,
        "test-un-r999-rev2",
        registry=reg,
        blob_store=FilesystemBlobStore(tmp_path / "b"),
        embedder=emb,
        repo_root=tmp_path,
    )
    assert out.status == "SUCCEEDED"
    return emb


def test_reranker_failure_degrades_not_fails(corpus, db_session) -> None:  # type: ignore[no-untyped-def]
    svc = RetrievalService(embedder=corpus, reranker=ExplodingReranker(), config=RetrievalConfig(min_shared_terms=1))
    r = svc.search(db_session, "thorax compression criterion limit", k=3)
    assert r.bundle.evidence, "fused order must still produce evidence"
    assert r.versions["degraded"] == ["reranker_failed:RuntimeError"]
    assert all(e.ranks.rerank_score is None for e in r.bundle.evidence)


def test_embedding_failure_is_loud(corpus, db_session, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.api.routes import health
    from safety_assistant.api.routes import query as q

    svc = RetrievalService(embedder=ExplodingEmbedder(), config=RetrievalConfig(min_shared_terms=1))
    q._retrieval.cache_clear()
    monkeypatch.setattr(q, "_retrieval", lambda: svc)
    monkeypatch.setattr(health, "get_embedding_provider", lambda: ExplodingEmbedder().embed_query("x"))
    from safety_assistant.api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    ready = client.get("/health/ready")
    assert ready.status_code == 503 and ready.json()["deps"]["embeddings"]["ok"] is False
    r = client.post("/api/v1/search", json={"query": "thorax compression criterion limit"})
    assert r.status_code == 500 and r.headers.get("x-request-id")  # never a silent fake vector


def test_database_down_flips_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    from safety_assistant.api.routes import health

    def boom() -> dict[str, object]:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(health, "_db_check", boom)
    from safety_assistant.api.main import app

    client = TestClient(app)
    r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["deps"]["database"]["ok"] is False and "ConnectionError" in r.json()["deps"]["database"]["error"]
    assert client.get("/health/live").status_code == 200  # liveness is process-only


def test_search_does_not_block_the_event_loop(corpus, db_session, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """A slow synchronous retrieval must run in the threadpool: a concurrent
    /health/live must still be answered while /search is in flight."""
    from safety_assistant.api.main import app
    from safety_assistant.api.routes import query as q

    class Slow(RetrievalService):
        def search(self, *a, **k):  # type: ignore[no-untyped-def]
            time.sleep(1.0)
            return super().search(*a, **k)

    q._retrieval.cache_clear()
    monkeypatch.setattr(q, "_retrieval", lambda: Slow(embedder=corpus, config=RetrievalConfig(min_shared_terms=1)))

    async def run() -> tuple[float, int]:
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            search = asyncio.create_task(c.post("/api/v1/search", json={"query": "thorax compression"}))
            await asyncio.sleep(0.2)
            t0 = time.perf_counter()
            live = await c.get("/health/live")
            dt = time.perf_counter() - t0
            resp = await search
            assert resp.status_code == 200
            return dt, live.status_code

    dt, status = asyncio.run(run())
    assert status == 200 and dt < 0.5, f"liveness took {dt:.2f}s while search was running"
