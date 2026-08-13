"""Application settings and SQLAlchemy engine/session wiring.

Single source of runtime configuration (`Settings`), env-var driven per
`ENVIRONMENT_SETUP.md` §8. Defaults match `docker-compose.yml` /
`.env.example` so local dev works without a `.env` file.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://passive_safety:change_me@localhost:5433/passive_safety"

    llm_provider: str = "freellmapi"
    llm_base_url: str = "http://localhost:3001/v1"
    llm_api_key: str = ""
    llm_model: str = ""

    knowledge_root: pathlib.Path = pathlib.Path("./knowledge")
    data_root: pathlib.Path = pathlib.Path("./data")

    log_level: str = "INFO"

    # apps/web's dev server (docs/ADR/0004: port 3010, not the common 3000
    # default, to avoid colliding with a sibling project on this host).
    # Override via env as a JSON array, e.g. CORS_ORIGINS='["https://example.com"]'
    # (pydantic-settings' default parsing for list-typed fields).
    cors_origins: list[str] = ["http://localhost:3010", "http://127.0.0.1:3010"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, future=True)


def get_session() -> Iterator[Session]:
    """FastAPI-dependency-shaped session factory: `Depends(get_session)`."""
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session
