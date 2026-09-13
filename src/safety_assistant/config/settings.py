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
    # cross_encoder measured 2026-09-12: MRR 0.713 → 0.808 on regulatory_v2 with rerank_top_n=12 (docs/evaluation.md)
    reranker: RerankerName = "cross_encoder"

    llm_provider: LLMProviderName = "none"
    llm_base_url: str = "http://localhost:3001/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 30.0
    # Data classes the configured LLM provider is cleared to see (explicit policy,
    # never inferred from a model name). CONFIDENTIAL evidence with a PUBLIC-only
    # provider degrades to evidence-only mode.
    llm_data_classes: list[str] = ["PUBLIC"]

    # Hosts the SSRF-safe fetcher may download from (https only, public IPs only).
    fetch_allowed_hosts: list[str] = ["unece.org", "www.unece.org"]

    # Per-principal request budget for /ask and /search (in-process token bucket).
    rate_limit_per_minute: int = 60

    # Retrieval candidate sizes (CLAUDE.md §8 starting points; tune from evals only).
    retrieval_dense_top_k: int = 30
    retrieval_sparse_top_k: int = 30
    retrieval_final_k: int = 10
    retrieval_dense_weight: float = 0.75  # RRF leg weights; tuned on evals/datasets (see docs/evaluation.md)
    retrieval_sparse_weight: float = 1.0
    retrieval_rerank_top_n: int | None = 12  # cap for expensive rerankers (cross_encoder); None = all
    retrieval_rerank_policy: Literal["always", "adaptive"] = "always"  # see docs/evaluation.md "adaptive reranking"

    # Ingestion resource limits (CLAUDE.md §14).
    ingest_max_file_bytes: int = 200 * 1024 * 1024
    ingest_max_pages: int = 5000
    # Untrusted-upload boundaries: malware scanning (clamd) and OCR for scanned pages are optional
    # adapters; `none` keeps the stage explicit rather than silently absent.
    malware_scanner: Literal["none", "clamav"] = "none"
    clamav_host: str = "localhost"
    clamav_port: int = 3310
    ocr_provider: Literal["none", "tesseract"] = "none"
    ocr_language: str = "eng"

    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3010", "http://127.0.0.1:3010"]

    # Auth (M11). Empty in development means "anonymous viewer"; production
    # refuses to start without an issuer or an API-key set.
    auth_mode: Literal["none", "api_key", "oidc"] = "none"
    api_keys: dict[str, str] = {}  # key -> role, dev/test convenience
    # OIDC (auth_mode=oidc for bearer tokens; the browser flow needs the client settings below).
    oidc_issuer: str = ""
    oidc_audience: str = ""  # expected `aud` of bearer access tokens (API-to-API); defaults to the client id
    oidc_client_id: str = ""
    oidc_client_secret: str = ""  # empty for public clients (PKCE only)
    oidc_redirect_uri: str = ""  # e.g. https://app.example.com/api/v1/auth/oidc/callback
    oidc_scopes: str = "openid profile email"
    # Role granted in the default organization on first sign-in ("" = no membership until an admin adds one).
    oidc_default_role: str = ""
    frontend_url: str = "http://localhost:3010"  # post-login redirect base (same origin as the API in production)
    # Browser sessions (ADR-0029 §6): HS256 JWT in an HttpOnly cookie signed with this secret.
    session_secret: str = ""
    session_ttl_hours: int = 12
    # Dev-only login as a seeded user (no password). Production refuses it.
    dev_login_enabled: bool = False

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
        if self.dev_login_enabled:
            problems.append("DEV_LOGIN_ENABLED bypasses the identity provider")
        if len(self.session_secret) < 32:
            problems.append("SESSION_SECRET must be at least 32 characters")
        if any(o.strip() in ("*", "null") for o in self.cors_origins):
            problems.append("CORS_ORIGINS must be an explicit allowlist (credentials are allowed)")
        if problems:
            raise ValueError("refusing to start in production: " + "; ".join(problems))
        return self

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"


@lru_cache
def get_settings() -> Settings:
    return Settings()
