"""Phase-0 environment smoke test.

Verifies the core Phase-0 dependencies import cleanly and that the
`packages/` tree is importable. No application logic is exercised here —
see `IMPLEMENTATION_PLAN.md` Phase Gates for the real gates.
"""

import importlib


def test_core_dependencies_import() -> None:
    for module in (
        "fastapi",
        "pydantic",
        "pydantic_settings",
        "sqlalchemy",
        "alembic",
        "psycopg",
        "pgvector",
        "numpy",
        "pandas",
        "pyarrow",
        "duckdb",
    ):
        importlib.import_module(module)


def test_repository_layout_exists() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for expected in (
        "packages/domain",
        "packages/analysis",
        "packages/ingestion",
        "packages/retrieval",
        "packages/agent",
        "knowledge/00_registry",
        "docs/ADR",
    ):
        assert (root / expected).is_dir(), f"missing {expected}"
