"""Summary-augmented chunking end to end on the synthetic corpus: summary generated once per
version and cached, failure fallback + retry, the sac_v1 index built beside the baseline,
hard-negative retrieval (lateral vs. synthetic-collision regulation), deterministic version
filtering unaffected, evidence never carrying the summary, and idempotent/resumable reindexing."""

from __future__ import annotations

import datetime
import pathlib

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from safety_assistant.config import get_settings
from safety_assistant.contextualization import SAC_COMPACT_REPRESENTATION, SAC_REPRESENTATION
from safety_assistant.contextualization.reindex import reindex_sac, sac_coverage
from safety_assistant.ingestion.fetch import FilesystemBlobStore
from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.persistence.models import Chunk, ChunkEmbedding, DocumentSummary, RegulationVersion
from safety_assistant.providers.embeddings.hashing import HashingEmbeddingProvider
from safety_assistant.providers.llm import LLMBadRequest, LLMMessage, LLMResponse
from safety_assistant.retrieval import RetrievalService, ScopeFilter
from safety_assistant.retrieval.service import RetrievalConfig
from safety_assistant.retrieval.sparse import invalidate_cache
from tests.conftest import requires_db
from tests.support.minireg import registry_for

pytestmark = requires_db


class _Summarizer:
    """Deterministic stand-in for the summary model: echoes the document's own scope sentence,
    so the summary carries exactly the discriminating term the document states."""

    name = "mock"
    model = "mock-summarizer"
    calls = 0

    def generate(self, messages: list[LLMMessage], *, schema=None, temperature=0.0, max_tokens=1024):  # type: ignore[no-untyped-def]
        self.calls += 1
        excerpt = messages[-1].content.split("<document_excerpt>", 1)[1]
        scope = next((ln for ln in excerpt.splitlines() if ln.startswith("This Regulation applies")), "")
        text = f"Retrieval summary for a synthetic test regulation. {scope} It defines performance criteria."
        return LLMResponse(content=text, model=self.model, provider=self.name, usage={"total_tokens": 30})


class _Down:
    name = "mock"
    model = "mock-summarizer"

    def generate(self, messages, *, schema=None, temperature=0.0, max_tokens=1024):  # type: ignore[no-untyped-def]
        raise LLMBadRequest("summary provider 400: context too long")


def _settings():  # type: ignore[no-untyped-def]
    return get_settings().model_copy(update={"sac_enabled": True})


def _ingest(session: Session, tmp: pathlib.Path, llm, keys: tuple[str, ...], emb, reg=None):  # type: ignore[no-untyped-def]
    reg = reg or registry_for(tmp, include_r998=True)  # build the PDFs once per test: fresh bytes → fresh sha
    store = FilesystemBlobStore(tmp / "b")
    outs = [
        ingest_source(
            session, k, registry=reg, blob_store=store, embedder=emb, repo_root=tmp, settings=_settings(), llm=llm
        )
        for k in keys
    ]
    invalidate_cache()
    return outs


ALL = ("test-un-r998-rev1", "test-un-r999-rev1", "test-un-r999-rev2")


def _sac_count(session: Session, representation: str = SAC_COMPACT_REPRESENTATION) -> int:
    """Ingestion and `reindex` build the compact representation (the recommended one) by default."""
    return (
        session.scalar(select(func.count(ChunkEmbedding.id)).where(ChunkEmbedding.representation == representation))
        or 0
    )


def test_summary_once_per_version_cached_and_indexed(clean_db, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    llm = _Summarizer()
    emb = HashingEmbeddingProvider(384)
    reg = registry_for(tmp_path, include_r998=True)
    outs = _ingest(db_session, tmp_path, llm, ALL, emb, reg)
    assert [o.status for o in outs] == ["SUCCEEDED"] * 3
    assert llm.calls == 3  # one summary per document version, not per chunk
    rows = db_session.scalars(select(DocumentSummary)).all()
    assert len(rows) == 3 and {r.status for r in rows} == {"READY"}
    assert len({r.cache_key for r in rows}) == 3  # different artifacts → different keys
    chunks = db_session.scalars(select(Chunk)).all()
    assert all(c.retrieval_text and c.retrieval_text.endswith(c.content) for c in chunks)
    assert all("DOCUMENT SUMMARY" in c.retrieval_text for c in chunks)
    assert _sac_count(db_session) == len(chunks)
    n_content = db_session.scalar(
        select(func.count(ChunkEmbedding.id)).where(ChunkEmbedding.representation == "content")
    )
    assert n_content == len(chunks)

    # unchanged re-ingest: no new summaries, no new LLM calls, nothing re-embedded
    again = _ingest(db_session, tmp_path, llm, ALL, emb, reg)
    # R999 Rev.1 is SUPERSEDED by then and reprocesses (existing lifecycle rule); the ACTIVE ones are no-ops
    assert again[0].status == again[2].status == "SKIPPED_UNCHANGED"
    assert llm.calls == 3  # even the reprocessed version hits the summary cache
    # reindex over a finished corpus is a no-op
    outcomes = reindex_sac(lambda: Session(db_session.get_bind()), emb, llm, allowed_data_classes=["PUBLIC"])
    assert sum(o.embedded for o in outcomes) == 0 and all(o.summary_status == "READY" for o in outcomes)
    assert llm.calls == 3
    cov = sac_coverage(db_session, emb)
    assert cov["chunks"] == cov["sac_embedded"] == len(chunks)


def test_summary_failure_falls_back_and_can_be_retried(clean_db, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    emb = HashingEmbeddingProvider(384)
    (out,) = _ingest(db_session, tmp_path, _Down(), ("test-un-r998-rev1",), emb)
    assert out.status == "SUCCEEDED" and out.final_version_status == "ACTIVE"  # summary failure never blocks
    row = db_session.scalar(select(DocumentSummary))
    assert row is not None and row.status == "FAILED" and "400" in (row.error or "")
    chunk = db_session.scalar(select(Chunk))
    assert chunk is not None and chunk.retrieval_text is not None
    assert "SOURCE DOCUMENT" in chunk.retrieval_text and "DOCUMENT SUMMARY" not in chunk.retrieval_text
    assert _sac_count(db_session) > 0  # never an empty searchable representation

    llm = _Summarizer()
    outcomes = reindex_sac(
        lambda: Session(db_session.get_bind()), emb, llm, allowed_data_classes=["PUBLIC"], retry_failed_summaries=True
    )
    assert outcomes[0].summary_status == "READY" and llm.calls == 1
    assert outcomes[0].embedded == outcomes[0].chunks  # retrieval text changed → every sac vector rebuilt
    db_session.expire_all()
    chunk = db_session.scalar(select(Chunk))
    assert chunk is not None and "DOCUMENT SUMMARY" in (chunk.retrieval_text or "")
    assert _sac_count(db_session) == db_session.scalar(select(func.count(Chunk.id)))


def test_confidential_document_is_not_sent_to_the_summary_provider(clean_db, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.contextualization import ensure_summary
    from safety_assistant.persistence.models import Regulation

    emb = HashingEmbeddingProvider(384)
    _ingest(db_session, tmp_path, None, ("test-un-r998-rev1",), emb)
    reg = db_session.scalar(select(Regulation))
    ver = db_session.scalar(select(RegulationVersion))
    assert reg is not None and ver is not None
    reg.data_class = "CONFIDENTIAL"
    llm = _Summarizer()
    row = ensure_summary(db_session, ver, reg, llm, allowed_data_classes=["PUBLIC"], retry_failed=True)
    assert row.status == "SKIPPED" and llm.calls == 0


def _service(emb, representation: str) -> RetrievalService:  # type: ignore[no-untyped-def]
    cfg = RetrievalConfig(min_shared_terms=1, use_dense=False, use_reranker=False, representation=representation)
    return RetrievalService(embedder=emb, config=cfg)


def test_hard_negative_sac_prefers_document_stating_the_scope(clean_db, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """R998 (lateral) and R999 (synthetic collision) both state a 42 mm deflection limit in
    near-identical wording. Only R998's *document* says "lateral"; the clause chunk itself does not."""
    emb = HashingEmbeddingProvider(384)
    _ingest(db_session, tmp_path, _Summarizer(), ALL, emb)
    q = "42 mm deflection limit in the lateral collision regulation"
    sac = _service(emb, SAC_REPRESENTATION).search(db_session, q, k=5).bundle.evidence
    assert sac and sac[0].regulation_key == "UN-R998"
    assert "3.1.2" in sac[0].section_path or "3.1" in sac[0].section_path
    base = _service(emb, "content").search(db_session, q, k=5).bundle.evidence
    assert base  # the baseline still answers; whether it picks R998 first is what the DRM metric measures


def test_version_filter_stays_deterministic_under_sac(clean_db, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    emb = HashingEmbeddingProvider(384)
    _ingest(db_session, tmp_path, _Summarizer(), ALL, emb)
    svc = _service(emb, SAC_REPRESENTATION)
    q = "thorax compression limit"
    current = svc.search(db_session, q, k=5).bundle.evidence
    assert current and all(e.version_label.startswith("Rev.2") for e in current if e.regulation_key == "UN-R999")
    historical = svc.search(db_session, q, scope=ScopeFilter(as_of=datetime.date(2020, 6, 1)), k=5).bundle.evidence
    assert historical and all(e.version_label.startswith("Rev.1") for e in historical if e.regulation_key == "UN-R999")
    assert any("42 mm" in e.content for e in historical if e.regulation_key == "UN-R999")


def test_evidence_from_sac_index_is_original_chunk_text(clean_db, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    emb = HashingEmbeddingProvider(384)
    _ingest(db_session, tmp_path, _Summarizer(), ALL, emb)
    ev = _service(emb, SAC_REPRESENTATION).search(db_session, "lateral collision rib deflection", k=5).bundle.evidence
    assert ev
    for e in ev:
        chunk = db_session.get(Chunk, e.chunk_id)
        assert chunk is not None and e.content == chunk.content
        assert "DOCUMENT SUMMARY" not in e.content and "SOURCE DOCUMENT" not in e.content
        assert "Retrieval summary" not in e.content


def test_reindex_resumes_after_a_failed_version(clean_db, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    emb = HashingEmbeddingProvider(384)
    # ingest without SAC, then build the index with `reindex` — the migration path for an existing corpus
    reg = registry_for(tmp_path, include_r998=True)
    store = FilesystemBlobStore(tmp_path / "b")
    for k in ALL:
        out = ingest_source(db_session, k, registry=reg, blob_store=store, embedder=emb, repo_root=tmp_path)
        assert out.status == "SUCCEEDED"
    assert _sac_count(db_session) == 0

    class _FlakyEmbedder(HashingEmbeddingProvider):
        failures = 1

        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            if self.failures:
                self.failures -= 1
                raise RuntimeError("embedding backend hiccup")
            return super().embed_documents(texts)

    flaky = _FlakyEmbedder(384)
    first = reindex_sac(lambda: Session(db_session.get_bind()), flaky, _Summarizer(), allowed_data_classes=["PUBLIC"])
    assert sum(1 for o in first if o.error) == 1 and sum(1 for o in first if not o.error) == 2
    partial = _sac_count(db_session)
    assert 0 < partial < db_session.scalar(select(func.count(Chunk.id)))
    second = reindex_sac(lambda: Session(db_session.get_bind()), flaky, _Summarizer(), allowed_data_classes=["PUBLIC"])
    assert not any(o.error for o in second)
    assert sum(o.embedded for o in second) == db_session.scalar(select(func.count(Chunk.id))) - partial
    cov = sac_coverage(db_session, emb)
    assert cov["chunks"] == cov["sac_embedded"]
    third = reindex_sac(lambda: Session(db_session.get_bind()), flaky, _Summarizer(), allowed_data_classes=["PUBLIC"])
    assert sum(o.embedded for o in third) == 0


def test_compact_representation_reindex_and_retrieval(clean_db, db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.contextualization import SAC_COMPACT_REPRESENTATION

    emb = HashingEmbeddingProvider(384)
    _ingest(db_session, tmp_path, _Summarizer(), ALL, emb)
    outcomes = reindex_sac(
        lambda: Session(db_session.get_bind()),
        emb,
        _Summarizer(),
        allowed_data_classes=["PUBLIC"],
        representation=SAC_COMPACT_REPRESENTATION,
    )
    assert not any(o.error for o in outcomes)
    cov = sac_coverage(db_session, emb, SAC_COMPACT_REPRESENTATION)
    assert cov["chunks"] == cov["sac_embedded"] == db_session.scalar(select(func.count(Chunk.id)))
    invalidate_cache()
    ev = (
        _service(emb, SAC_COMPACT_REPRESENTATION)
        .search(db_session, "42 mm deflection limit in the lateral collision regulation", k=5)
        .bundle.evidence
    )
    assert ev and ev[0].regulation_key == "UN-R998"
    for e in ev:
        chunk = db_session.get(Chunk, e.chunk_id)
        assert chunk is not None and e.content == chunk.content
