"""Root pytest configuration: run the whole suite against a dedicated
`passive_safety_test` database (same Postgres server, port 5433) instead of
the shared dev database `scripts/generate_synthetic_dataset.py` loads into.

Resolves the "no separate test database yet" gap documented repeatedly
across tests/conftest.py, tests/domain/test_migrations.py, and several
commit messages.

IMPORTANT: this all runs at module import time, not inside a
`pytest_configure` hook. pytest imports every conftest.py it can find
(root, tests/, tests/domain/, ...) *before* calling any `pytest_configure`
hook, because it has to import a conftest module before it can even know
what hooks that module defines. `tests/conftest.py` calls
`get_engine()`/`get_settings()` (both `@lru_cache`) at its own module level
(`requires_db = pytest.mark.skipif(not _database_reachable(), ...)`), so by
the time a `pytest_configure` hook would run, that cache is already
poisoned with whatever `DATABASE_URL` was set to before this file got a
chance to change it. Running as plain module-level code instead means this
executes as soon as *this* file is imported — which pytest does before
descending into tests/, since this file is in the rootdir.

Self-seeding: if the test database is empty, this also runs the same
synthetic-dataset generation, knowledge ingestion, and indexing steps the
`scripts/*.py` do — so `uv run pytest` is reproducible from a clean database
with nothing more than Postgres running, not a private setup ritual.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

import psycopg
from psycopg import sql

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (this file lives in tests/legacy/)
sys.path.insert(0, str(ROOT))

_TEST_DB_NAME = "passive_safety_test"
_DEFAULT_TEST_DB_URL = f"postgresql+psycopg://passive_safety:change_me@localhost:5433/{_TEST_DB_NAME}"
_ADMIN_DB_URL = "postgresql://passive_safety:change_me@localhost:5433/passive_safety"


def _plain_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://")


def _ensure_test_database_exists() -> bool:
    """Returns True if Postgres itself is reachable at all."""
    try:
        with psycopg.connect(_ADMIN_DB_URL, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (_TEST_DB_NAME,)).fetchone()
            if not exists:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(_TEST_DB_NAME)))
        return True
    except psycopg.OperationalError:
        return False


def _ensure_vector_extension(database_url: str) -> None:
    with psycopg.connect(_plain_url(database_url), autocommit=True, connect_timeout=3) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def _run_migrations() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "packages" / "domain" / "migrations"))
    command.upgrade(cfg, "head")


def _load_script(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(f"_conftest_script_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_if_empty() -> None:
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    from packages.domain.core import SimulationRun
    from packages.domain.db import get_engine
    from packages.domain.knowledge import DocumentChunk, Embedding

    with Session(get_engine()) as session:
        run_count = session.scalar(select(func.count()).select_from(SimulationRun)) or 0
        chunk_count = session.scalar(select(func.count()).select_from(DocumentChunk)) or 0
        embedding_count = session.scalar(select(func.count()).select_from(Embedding)) or 0

    if run_count == 0:
        _load_script("generate_synthetic_dataset").main()
    if chunk_count == 0:
        _load_script("ingest_documents").main()
    if embedding_count == 0:
        _load_script("index_knowledge").main()


def _setup() -> None:
    # Forced, not setdefault: tests/conftest.py (the rebuilt suite) exports its own
    # DATABASE_URL first. Run this suite separately: uv run pytest tests/legacy -o addopts=""
    os.environ["DATABASE_URL"] = os.environ.get("LEGACY_TEST_DATABASE_URL", _DEFAULT_TEST_DB_URL)
    database_url = os.environ["DATABASE_URL"]

    if not _ensure_test_database_exists():
        return  # Postgres unreachable — tests/conftest.py's requires_db will skip everything

    try:
        _ensure_vector_extension(database_url)
        _run_migrations()
        _seed_if_empty()
    except Exception as exc:  # pragma: no cover - defensive: never let setup crash collection
        print(f"[conftest] test database setup failed, tests requiring it will skip or fail: {exc}")


_setup()  # module-level: see the docstring above for why this can't be a pytest_configure hook


# ---- fixtures formerly in tests/conftest.py ----------------------------------
from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402
import sqlalchemy as sa  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from packages.domain.db import get_engine, get_settings  # noqa: E402


def _database_reachable() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _database_reachable(),
    reason=f"Postgres not reachable at {get_settings().database_url} "
    "(run: docker compose up -d postgres && uv run alembic upgrade head)",
)


@pytest.fixture
def session() -> Iterator[Session]:
    """A session wrapped in a transaction that's always rolled back.

    `join_transaction_mode="create_savepoint"` (SQLAlchemy 2.0) runs the
    session inside a SAVEPOINT rather than the outer transaction directly,
    and transparently restarts it if the ORM's own transaction ends (e.g. a
    test deliberately triggers and catches an IntegrityError to prove a
    constraint is enforced). Without this, such a test leaves the
    connection's outer transaction aborted, and this fixture's own teardown
    rollback produces a benign-but-confusing SAWarning.
    """
    engine = get_engine()
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
