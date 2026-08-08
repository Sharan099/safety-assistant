"""Centralized typed configuration — single source of truth for all settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    APP_NAME: str = "UNECE Passive Safety Assistant"
    API_PREFIX: str = "/api/v1"
    ROOT_DIR: Path = Path(__file__).resolve().parents[1]
    PDF_DIR: Path = ROOT_DIR / "data" / "pdfs"
    DOCS_DIR: Path = ROOT_DIR / "data" / "docs"

    # Database
    DATABASE_URL: str = (
        "postgresql://postgres:postgrespassword@localhost:5432/safety_registry"
    )

    # Embeddings + reranker
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_MAX_PASSAGE_CHARS: int = 256
    ENABLE_RERANKER: bool = True
    OMP_NUM_THREADS: int = Field(default_factory=lambda: max(1, os.cpu_count() or 4))

    # Retrieval tuning
    DEFAULT_TOP_K: int = 8
    LLM_CONTEXT_TOP_K: int = 8
    FUSION_POOL_BASE: int = 10
    FUSION_POOL_DEFINITION: int = 20
    FUSION_POOL_COMPARISON: int = 20
    ENABLE_REGULATION_PREFILTER: bool = False
    ENABLE_ADAPTIVE_CONTEXT: bool = True
    ENABLE_CONTEXT_DEDUPE: bool = True
    STRUCTURAL_CITATIONS: bool = False
    ENABLE_MULTI_QUERY: bool = True
    MULTI_QUERY_COUNT: int = 2
    ENABLE_QUERY_DECOMPOSITION: bool = True
    MAX_CHUNK_CHARS: int = 2400
    ENABLE_RETRIEVAL_CACHE: bool = True
    RETRIEVAL_CACHE_SIZE: int = 256

    # Generation — Groq-only LLM router (failover across Groq models)
    GROQ_API_KEY: str = ""
    ENABLE_GATEWAY: bool = True
    MAX_OUTPUT_TOKENS: int = 768
    GATEWAY_PRIMARY_MODEL: str = "groq"
    GATEWAY_FALLBACK_CHAIN: str = "groq,groq_fast,groq_instant"
    LLM_ROUTER_CHAIN: str = "groq_qwen3,groq_70b,groq_instant"
    LLM_ROUTER_MAX_RETRIES: int = 4
    LLM_ROUTER_COOLDOWN_SEC: float = 60.0
    LLM_ROUTER_TIMEOUT_SEC: float = 60.0
    GROUNDING_MIN_CONFIDENCE: float = 0.25

    # Ingest
    PARENT_CHILD_CHUNKING: bool = True
    INCREMENTAL_INGEST: bool = True
    SUPPORTED_EXTENSIONS: str = ".pdf,.md,.txt"

    # Security
    RATE_LIMIT_PER_MINUTE: int = 30
    MAX_QUERY_CHARS: int = 2000

    # Observability
    DEBUG_TIMING: bool = False
    ENABLE_PROMETHEUS: bool = False
    LOG_LEVEL: str = "INFO"

    @field_validator("OMP_NUM_THREADS", mode="before")
    @classmethod
    def _omp_threads(cls, v: object) -> int:
        if v is None or v == "":
            return max(1, os.cpu_count() or 4)
        return max(1, int(v))

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(ext.strip() for ext in self.SUPPORTED_EXTENSIONS.split(",") if ext.strip())


settings = Settings()
os.makedirs(settings.PDF_DIR, exist_ok=True)
os.makedirs(settings.DOCS_DIR, exist_ok=True)
