"""Test configuration for the safety_assistant suite.

Profile: APP_ENV=test, hashing embeddings (no model download, deterministic),
mock LLM, dedicated `safety_assistant_test` database migrated to head.
Integration/e2e tests that need PostgreSQL are skipped when it is unreachable.
Tests under tests/legacy have their own conftest and the pre-rebuild database.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://passive_safety:change_me@localhost:5433/safety_assistant_test"
)
ADMIN_DB_URL = os.environ.get(
    "ADMIN_DATABASE_URL", "postgresql://passive_safety:change_me@localhost:5433/passive_safety"
)

os.environ.update(
    APP_ENV="test",
    DATABASE_URL=TEST_DB_URL,
    EMBEDDING_PROVIDER="hashing",
    LLM_PROVIDER="mock",
    AUTH_MODE="none",
    ARTIFACT_STORE_URI=f"file://{(ROOT / 'data' / 'artifacts-test').as_posix()}",
)


def _postgres_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(ADMIN_DB_URL, connect_timeout=3):
            return True
    except Exception:  # noqa: BLE001
        return False


POSTGRES_AVAILABLE = _postgres_available()
requires_db = pytest.mark.skipif(
    not POSTGRES_AVAILABLE, reason="PostgreSQL not reachable (docker compose up -d postgres)"
)


def _bootstrap_test_db() -> None:
    import psycopg
    from alembic import command
    from alembic.config import Config
    from psycopg import sql

    name = TEST_DB_URL.rsplit("/", 1)[-1]
    with psycopg.connect(ADMIN_DB_URL, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (name,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    with psycopg.connect(TEST_DB_URL.replace("postgresql+psycopg://", "postgresql://"), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cfg = Config(str(ROOT / "migrations" / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    command.upgrade(cfg, "head")


if POSTGRES_AVAILABLE:
    _bootstrap_test_db()


@pytest.fixture(scope="session")
def engine():  # type: ignore[no-untyped-def]
    if not POSTGRES_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    from safety_assistant.persistence import get_engine

    return get_engine()


@pytest.fixture
def db_session(engine) -> Iterator[object]:  # type: ignore[no-untyped-def]
    """Plain session; tests that write commit through the workflow, so use `clean_db`."""
    from sqlalchemy.orm import Session

    with Session(engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def clean_db(engine) -> None:  # type: ignore[no-untyped-def]
    """Truncate every application table (not alembic_version) and drop caches."""
    from sqlalchemy import text

    from safety_assistant.persistence import Base
    from safety_assistant.retrieval.sparse import invalidate_cache

    names = ", ".join(t.name for t in Base.metadata.sorted_tables)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
    invalidate_cache()
