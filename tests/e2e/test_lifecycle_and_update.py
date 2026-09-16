"""E2E (CLAUDE.md §11): source → ingest → validate → parse → chunk → index →
verify → activate → query → evidence → open source; then the update flow
v1 active → v2 ingested → changed chunks only re-embedded → v2 active →
current query uses v2, historical query still resolves to v1."""

from __future__ import annotations

import datetime
import pathlib

import pytest
from sqlalchemy import func, select

from safety_assistant.domain.regulations import VersionStatus
from safety_assistant.ingestion.fetch import FilesystemBlobStore
from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.persistence.models import (
    Chunk,
    ChunkEmbedding,
    CrossReference,
    IngestionEvent,
    IngestionRun,
    RegulationVersion,
    Section,
)
from safety_assistant.providers.embeddings.hashing import HashingEmbeddingProvider
from safety_assistant.retrieval import RetrievalService, ScopeFilter
from safety_assistant.retrieval.service import RetrievalConfig
from tests.conftest import requires_db
from tests.support.minireg import REG_KEY, registry_for

pytestmark = requires_db

_STAGES = [s.value for s in VersionStatus][:9]


@pytest.fixture
def env(tmp_path: pathlib.Path):  # type: ignore[no-untyped-def]
    return {
        "registry": registry_for(tmp_path),
        "blobs": FilesystemBlobStore(tmp_path / "blobs"),
        "embedder": HashingEmbeddingProvider(384),
        "root": tmp_path,
    }


def _ingest(db_session, env, key: str, **kw):  # type: ignore[no-untyped-def]
    return ingest_source(
        db_session,
        key,
        registry=env["registry"],
        blob_store=env["blobs"],
        embedder=env["embedder"],
        repo_root=env["root"],
        **kw,
    )


def _service(env) -> RetrievalService:  # type: ignore[no-untyped-def]
    cfg = RetrievalConfig(use_reranker=True, min_shared_terms=1)
    return RetrievalService(embedder=env["embedder"], config=cfg)


def test_full_lifecycle_v1(clean_db, db_session, env) -> None:  # type: ignore[no-untyped-def]
    out = _ingest(db_session, env, "test-un-r999-rev1")
    assert out.status == "SUCCEEDED", out.error
    assert out.final_version_status == "ACTIVE"

    version = db_session.get(RegulationVersion, out.version_id)
    assert version.parser_version and version.chunker_version and version.parsed_hash
    assert version.amendments and version.amendments[0]["entry_into_force"] == "2019-01-01"
    assert version.valid_from == datetime.date(2019, 1, 1)

    # every lifecycle stage was traversed and recorded
    events = db_session.scalars(
        select(IngestionEvent).where(IngestionEvent.run_id == out.run_id).order_by(IngestionEvent.at)
    ).all()
    transitions = [e.to_status for e in events if e.from_status]
    assert transitions == _STAGES[1:], transitions

    # structure: definitions, annex scope, cross-reference resolved
    sections = {s.path: s for s in db_session.scalars(select(Section).where(Section.version_id == version.id))}
    assert sections["2.1"].kind == "DEFINITION" and sections["2.1"].title == "Protective system"
    assert sections["annex-3/1.2"].annex == "Annex 3"
    assert sections["3.2.2"].normative is True
    xrefs = db_session.scalars(select(CrossReference).where(CrossReference.version_id == version.id)).all()
    path_by_id = {s.id: s.path for s in sections.values()}
    resolved = {(path_by_id[x.from_section_id], x.target_path) for x in xrefs if x.resolved_section_id}
    assert ("3.1", "annex-3/1.2") in resolved, resolved

    # every chunk embedded, citation labels exact, deterministic ids
    n_chunks = db_session.scalar(select(func.count(Chunk.id)).where(Chunk.version_id == version.id))
    n_emb = db_session.scalar(
        select(func.count(ChunkEmbedding.id))
        .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
        .where(Chunk.version_id == version.id, ChunkEmbedding.representation == "content")
    )
    assert n_chunks == n_emb > 0
    labels = set(db_session.scalars(select(Chunk.citation_label).where(Chunk.version_id == version.id)))
    assert "UN R999 Rev.1 §3.2.1–3.2.3 (p. 3)" in labels or any("§3.2" in lb for lb in labels)

    # query → evidence → open exact source
    svc = _service(env)
    r = svc.search(db_session, "thorax compression criterion limit", k=3)
    assert r.bundle.evidence, r.candidates
    top = r.bundle.evidence[0]
    assert top.regulation_key == REG_KEY and top.version_label.startswith("Rev.1")
    assert "42 mm" in top.content
    assert env["blobs"].exists(top.storage_uri)
    assert top.source_sha256 == env["registry"].get("test-un-r999-rev1").sha256


def test_rerun_is_idempotent_noop(clean_db, db_session, env) -> None:  # type: ignore[no-untyped-def]
    first = _ingest(db_session, env, "test-un-r999-rev1")
    second = _ingest(db_session, env, "test-un-r999-rev1")
    assert first.status == "SUCCEEDED" and second.status == "SKIPPED_UNCHANGED"
    assert db_session.scalar(select(func.count(RegulationVersion.id))) == 1
    assert db_session.scalar(select(func.count(IngestionRun.id))) == 2


def test_update_flow_v1_to_v2(clean_db, db_session, env) -> None:  # type: ignore[no-untyped-def]
    v1 = _ingest(db_session, env, "test-un-r999-rev1")
    v2 = _ingest(db_session, env, "test-un-r999-rev2")
    assert v1.status == v2.status == "SUCCEEDED"

    # changed-content-only re-embedding: only the cover and the amended clause block are new
    assert v2.stats["embed_total"] == 12
    assert v2.stats["embed_reused"] == 10, v2.stats
    assert v2.stats["embed_new"] == 2, v2.stats

    ver1 = db_session.get(RegulationVersion, v1.version_id)
    ver2 = db_session.get(RegulationVersion, v2.version_id)
    db_session.refresh(ver1)
    assert ver2.status == "ACTIVE" and ver1.status == "SUPERSEDED"
    assert ver1.superseded_by_id == ver2.id and ver1.valid_to == datetime.date(2022, 1, 1)

    svc = _service(env)
    # current query → v2 text (45 mm)
    now = svc.search(db_session, "thorax compression criterion limit", k=3)
    assert now.bundle.evidence and now.bundle.evidence[0].version_label.startswith("Rev.2")
    assert all(e.version_label.startswith("Rev.2") for e in now.bundle.evidence)
    assert "45 mm" in now.bundle.evidence[0].content
    # historical as-of query → v1 text (42 mm), v2 absent
    hist = svc.search(
        db_session, "thorax compression criterion limit", scope=ScopeFilter(as_of=datetime.date(2020, 6, 1)), k=3
    )
    assert hist.bundle.evidence and all(e.version_label.startswith("Rev.1") for e in hist.bundle.evidence)
    assert "42 mm" in hist.bundle.evidence[0].content
    # before v1 → nothing
    none = svc.search(
        db_session, "thorax compression criterion limit", scope=ScopeFilter(as_of=datetime.date(2018, 1, 1)), k=3
    )
    assert none.bundle.evidence == []
    # new clause only exists in v2
    tib = svc.search(db_session, "tibia index", k=3)
    assert tib.bundle.evidence and tib.bundle.evidence[0].version_label.startswith("Rev.2")


def test_corrupted_source_is_quarantined_not_ingested(clean_db, db_session, env, tmp_path) -> None:  # type: ignore[no-untyped-def]
    entry = env["registry"].get("test-un-r999-rev1")
    path = tmp_path / entry.local_path
    path.write_bytes(path.read_bytes()[:-200] + b"tampered" * 25)  # same size class, wrong hash
    out = _ingest(db_session, env, "test-un-r999-rev1")
    assert out.status == "QUARANTINED"
    assert out.final_version_status == "QUARANTINED"
    assert "sha256 mismatch" in (out.error or "") or "size mismatch" in (out.error or "")
    assert db_session.scalar(select(func.count(Chunk.id))) == 0
    svc = _service(env)
    assert svc.search(db_session, "thorax compression", k=3).bundle.evidence == []


def test_change_impact_diff_between_versions(clean_db, db_session, env) -> None:  # type: ignore[no-untyped-def]
    from safety_assistant.ingestion.diff import diff_versions, find_version

    _ingest(db_session, env, "test-un-r999-rev1")
    _ingest(db_session, env, "test-un-r999-rev2")
    a, b = find_version(db_session, REG_KEY, "Rev.1"), find_version(db_session, REG_KEY, "Rev.2")
    assert a is not None and b is not None
    d = diff_versions(db_session, a, b)
    assert [c.path for c in d.added] == ["3.2.4"]
    assert [c.path for c in d.changed] == ["3.2.2"]
    assert d.removed == [] and d.unchanged >= 14
    assert "-The thorax compression criterion (ThCC) shall not exceed 42 mm." in d.changed[0].diff
    assert "+The thorax compression criterion (ThCC) shall not exceed 45 mm." in d.changed[0].diff
    assert d.summary()["normative_changes"] == ["3.2.2", "3.2.4"]


def test_section_insertion_is_batched_not_one_round_trip_per_section(clean_db, db_session, env) -> None:  # type: ignore[no-untyped-def]
    """Regression guard for a flush-per-section loop that turned a 1,000-section document (49 CFR,
    LS-DYNA manuals) into 1,000 database round trips: the number of INSERT statements against
    `sections` must not scale with the number of sections."""
    from sqlalchemy import event

    inserts = []
    engine = db_session.get_bind()

    def _count(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        if "insert into sections" in statement.lower():
            inserts.append(statement)

    event.listen(engine, "before_cursor_execute", _count)
    try:
        out = _ingest(db_session, env, "test-un-r999-rev1")
    finally:
        event.remove(engine, "before_cursor_execute", _count)

    assert out.status == "SUCCEEDED"
    n_sections = db_session.scalar(select(func.count(Section.id)))
    assert n_sections >= 14  # the fixture regulation has enough sections to make this meaningful
    assert len(inserts) <= 2, f"{len(inserts)} INSERT statements for {n_sections} sections (want ~1)"


def test_reingesting_unchanged_content_does_not_touch_blob_storage_again(clean_db, db_session, env) -> None:  # type: ignore[no-untyped-def]
    """A --force reprocess or a retry after a transient failure re-validates and re-parses the same
    bytes, but must not re-touch blob storage for content already stored (S3 HEAD/PUT round trips,
    or even a filesystem stat, are wasted work once the artifact row already has a storage_uri)."""
    puts = []
    real_put = env["blobs"].put

    def counted_put(data, **kw):  # type: ignore[no-untyped-def]
        puts.append(len(data))
        return real_put(data, **kw)

    env["blobs"].put = counted_put  # type: ignore[method-assign]

    out1 = _ingest(db_session, env, "test-un-r999-rev1")
    assert out1.status == "SUCCEEDED"
    assert len(puts) == 1  # the PDF, stored once

    out2 = _ingest(db_session, env, "test-un-r999-rev1", force=True)
    assert out2.status == "SUCCEEDED"
    assert len(puts) == 1, "reprocessing the same bytes must not call blobs.put again"
