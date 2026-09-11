"""Runtime configuration — one env-driven `Settings` object.

`app_env` selects a profile:

- ``production``: fakes are refused at construction time (CLAUDE.md §2.3 /
  §5.4 — never silently use mock embeddings or fake LLM output). Readiness
  fails if the real embedding model cannot load.
- ``development``: real providers by default, fakes allowed if asked for.
- ``test``: fakes allowed; the test suite selects them explicitly.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["development", "test", "production"]
EmbeddingProviderName = Literal["fastembed", "hashing"]
RerankerName = Literal["heuristic", "cross_encoder", "none"]
LLMProviderName = Literal["openai_compatible", "mock", "none"]

# Fixed by the pgvector column type (migrations/versions/0001) and the HNSW
# index built on it. Changing the embedding model to another dimension is a
# schema migration, on purpose — see docs/ADR/0019.
EMBEDDING_DIMENSIONS = 384


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: AppEnv = "development"

    # PostgreSQL is the canonical metadata + retrieval store (docs/ADR/0001, 0019).
    database_url: str = "postgresql+psycopg://passive_safety:change_me@localhost:5433/safety_assistant"
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Object storage for immutable source artifacts / parser output. The
    # filesystem adapter is the development backend; production points at an
    # S3-compatible bucket (`s3://bucket/prefix`).
    artifact_store_uri: str = "file://./data/artifacts"
    knowledge_root: pathlib.Path = pathlib.Path("./knowledge")

    embedding_provider: EmbeddingProviderName = "fastembed"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker: RerankerName = "heuristic"

    llm_provider: LLMProviderName = "none"
    llm_base_url: str = "http://localhost:3001/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 30.0

    # Retrieval candidate sizes (CLAUDE.md §8 starting points; tune from evals only).
    retrieval_dense_top_k: int = 30
    retrieval_sparse_top_k: int = 30
    retrieval_final_k: int = 10

    # Ingestion resource limits (CLAUDE.md §14).
    ingest_max_file_bytes: int = 200 * 1024 * 1024
    ingest_max_pages: int = 5000

    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3010", "http://127.0.0.1:3010"]

    # Auth (M11). Empty in development means "anonymous viewer"; production
    # refuses to start without an issuer or an API-key set.
    auth_mode: Literal["none", "api_key", "oidc"] = "none"
    api_keys: dict[str, str] = {}  # key -> role, dev/test convenience
    oidc_issuer: str = ""
    oidc_audience: str = ""

    @model_validator(mode="after")
    def _production_forbids_fakes(self) -> Settings:
        if self.app_env != "production":
            return self
        problems = []
        if self.embedding_provider == "hashing":
            problems.append("EMBEDDING_PROVIDER=hashing is a test fake")
        if self.llm_provider == "mock":
            problems.append("LLM_PROVIDER=mock is a test fake")
        if self.auth_mode == "none":
            problems.append("AUTH_MODE=none leaves privileged endpoints open")
        if "change_me" in self.database_url:
            problems.append("DATABASE_URL still uses the development password")
        if problems:
            raise ValueError("refusing to start in production: " + "; ".join(problems))
        return self

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"


@lru_cache
def get_settings() -> Settings:
    return Settings()
