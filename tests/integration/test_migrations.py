"""Every migration must downgrade and re-upgrade against a database holding corpus rows
(ADR-0029 §3): the chain is exercised head → 0001 → head, and the corpus survives."""

from __future__ import annotations

import pathlib

from alembic import command
from alembic.config import Config
from sqlalchemy import select, text

from safety_assistant.ingestion.fetch import FilesystemBlobStore
from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.persistence.models import Chunk
from safety_assistant.providers.embeddings.hashing import HashingEmbeddingProvider
from tests.conftest import ROOT, TEST_DB_URL, requires_db
from tests.support.minireg import registry_for

pytestmark = requires_db


def _cfg() -> Config:
    cfg = Config(str(ROOT / "migrations" / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    return cfg


def test_downgrade_to_baseline_and_back_keeps_corpus(clean_db, db_session, engine, tmp_path: pathlib.Path) -> None:  # type: ignore[no-untyped-def]
    out = ingest_source(
        db_session,
        "test-un-r999-rev2",
        registry=registry_for(tmp_path, revisions=(2,)),
        blob_store=FilesystemBlobStore(tmp_path / "b"),
        embedder=HashingEmbeddingProvider(384),
        repo_root=tmp_path,
    )
    assert out.status == "SUCCEEDED"
    n_chunks = db_session.scalar(select(text("count(*)")).select_from(Chunk))
    db_session.close()

    command.downgrade(_cfg(), "0001")
    with engine.connect() as c:
        assert c.execute(text("select version_num from alembic_version")).scalar() == "0001"
        tables = {r[0] for r in c.execute(text("select tablename from pg_tables where schemaname='public'"))}
        assert "users" not in tables and "conversations" not in tables and "chunks" in tables
    command.upgrade(_cfg(), "head")
    with engine.connect() as c:
        assert c.execute(text("select count(*) from chunks")).scalar() == n_chunks
        tables = {r[0] for r in c.execute(text("select tablename from pg_tables where schemaname='public'"))}
        assert {"users", "organizations", "memberships", "workspaces", "conversations", "messages"} <= tables
