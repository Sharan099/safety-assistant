"""Data-class isolation and stale-version exclusion at the retrieval layer —
authorization is applied in SQL before ranking, not in the UI."""

from __future__ import annotations

import pathlib

import pytest
from sqlalchemy import update

from safety_assistant.ingestion.fetch import FilesystemBlobStore
from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.persistence.models import Regulation, RegulationVersion
from safety_assistant.providers.embeddings.hashing import HashingEmbeddingProvider
from safety_assistant.retrieval import RetrievalService, ScopeFilter
from safety_assistant.retrieval.service import RetrievalConfig
from tests.conftest import requires_db
from tests.support.minireg import REG_KEY, registry_for

pytestmark = requires_db


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
    return RetrievalService(embedder=emb, config=RetrievalConfig(min_shared_terms=1))


def test_confidential_regulation_is_invisible_to_public_scope(corpus, db_session) -> None:  # type: ignore[no-untyped-def]
    assert corpus.search(db_session, "thorax compression criterion", k=3).bundle.evidence
    db_session.execute(update(Regulation).where(Regulation.regulation_key == REG_KEY).values(data_class="CONFIDENTIAL"))
    db_session.commit()
    from safety_assistant.retrieval.sparse import invalidate_cache

    invalidate_cache()
    public = corpus.search(db_session, "thorax compression criterion", scope=ScopeFilter(data_classes=("PUBLIC",)), k=3)
    assert public.bundle.evidence == []
    engineer = corpus.search(
        db_session, "thorax compression criterion", scope=ScopeFilter(data_classes=("PUBLIC", "CONFIDENTIAL")), k=3
    )
    assert engineer.bundle.evidence


@pytest.mark.parametrize("stale", ["QUARANTINED", "VERIFIED", "INDEXED", "FAILED"])
def test_non_active_versions_are_never_retrievable(corpus, db_session, stale: str) -> None:  # type: ignore[no-untyped-def]
    db_session.execute(update(RegulationVersion).values(status=stale))
    db_session.commit()
    from safety_assistant.retrieval.sparse import invalidate_cache

    invalidate_cache()
    assert corpus.search(db_session, "thorax compression criterion", k=3).bundle.evidence == []
    assert (
        corpus.search(
            db_session, "thorax compression criterion", scope=ScopeFilter(include_superseded=True), k=3
        ).bundle.evidence
        == []
    )
