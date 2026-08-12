"""Backward-compatible re-export — fixtures now live in tests/conftest.py
(pytest fixtures are only visible to the directory they're defined in and
its subdirectories, so a directory-specific conftest can't serve sibling
packages like tests/ingestion or tests/api).
"""

from tests.conftest import requires_db, session

__all__ = ["requires_db", "session"]
