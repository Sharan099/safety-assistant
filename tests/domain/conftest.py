"""Shared fixtures for domain tests.

Requires the project's Postgres container (`docker compose up -d postgres`,
port 5433) with migrations applied (`uv run alembic upgrade head`). These
are integration tests against a real database, not mocks — BACKEND_SCHEMA.md
data invariants (FKs, uniqueness, JSONB) only mean something against real
Postgres/pgvector.

Known gap: there is no separate test database yet, so these tests share
Postgres with anything `scripts/generate_synthetic_dataset.py` has loaded.
Fixtures must therefore avoid colliding with real synthetic-benchmark rows
(e.g. use `__test` suffixes on unique keys) rather than assuming a clean DB.
A dedicated test DB is Phase-20 hardening work, not a Phase-6 blocker.
"""

from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from packages.domain.db import get_engine, get_settings


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
    """A session wrapped in a transaction that's always rolled back."""
    engine = get_engine()
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
