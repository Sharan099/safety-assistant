"""Shared fixtures for every integration test package under tests/.

Requires Postgres (`docker compose up -d postgres`, port 5433). These are
integration tests against a real database, not mocks — BACKEND_SCHEMA.md
data invariants (FKs, uniqueness, JSONB) only mean something against real
Postgres/pgvector.

The root `conftest.py` points the whole session at an isolated
`passive_safety_test` database (migrated and self-seeded automatically,
never the shared dev database) — see that file for how. Fixtures still
avoid colliding with the seeded synthetic-benchmark rows (e.g. `__test`
suffixes on unique keys) since that data is real and other tests depend on
it being there, but there's no risk of colliding with a developer's own
manual dev-DB experiments anymore.
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
