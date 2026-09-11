"""SQLAlchemy engine/session wiring. Sessions are request/unit-of-work scoped."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from safety_assistant.config import get_settings


@lru_cache
def get_engine() -> Engine:
    s = get_settings()
    return create_engine(s.database_url, pool_size=s.db_pool_size, max_overflow=s.db_max_overflow, pool_pre_ping=True)


def get_session() -> Iterator[Session]:
    """FastAPI-dependency-shaped: `Depends(get_session)`."""
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session
