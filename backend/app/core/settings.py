"""Application settings — imports shared model/path config from config.py."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    BM25_WEIGHT,
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    EMBEDDING_DOC_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_TRUST_REMOTE_CODE,
    ENABLE_RERANKER,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OUTPUT_DIR,
    RERANKER_MODEL,
    RRF_K,
    SEMANTIC_WEIGHT,
    SYSTEM_PROMPT,
    TOP_K_AFTER_RERANK,
    TOP_K_CHUNKS,
    TOP_K_RETRIEVE,
    TOP_K_VECTOR,
    VECTOR_SCORE_THRESHOLD,
)

__all__ = [
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOC_PREFIX",
    "EMBEDDING_TRUST_REMOTE_CODE",
    "RERANKER_MODEL",
    "ENABLE_RERANKER",
]

# Advanced retrieval stages
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
ENABLE_MULTI_QUERY = os.getenv("ENABLE_MULTI_QUERY", "true").lower() == "true"
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))
ENABLE_HARD_METADATA_FILTER = os.getenv("ENABLE_HARD_METADATA_FILTER", "true").lower() == "true"
ENABLE_METADATA_FILTER = os.getenv("ENABLE_METADATA_FILTER", "true").lower() == "true"
METADATA_BOOST = float(os.getenv("METADATA_BOOST", "0.5"))
ENABLE_PARENT_CHILD = os.getenv("ENABLE_PARENT_CHILD", "true").lower() == "true"

ENABLE_GROUNDING_GATE = os.getenv("ENABLE_GROUNDING_GATE", "true").lower() == "true"
GROUNDING_MIN_SEMANTIC = float(os.getenv("GROUNDING_MIN_SEMANTIC", "0.45"))
GROUNDING_MIN_RERANK_PROB = float(os.getenv("GROUNDING_MIN_RERANK_PROB", "0.5"))
REQUIRE_CITATIONS = os.getenv("REQUIRE_CITATIONS", "true").lower() == "true"

ABSTAIN_MESSAGE = os.getenv(
    "ABSTAIN_MESSAGE",
    "I don't know — I could not find this in the indexed regulation corpus. "
    "Try rephrasing, naming a specific regulation (e.g. UN R14, UN R16), or "
    "upload the relevant document to this chat.",
)

# Three mutually exclusive response states (Phase 1)
INJECTION_BLOCKED_MESSAGE = os.getenv(
    "INJECTION_BLOCKED_MESSAGE",
    "Request blocked.",
)
LOW_GROUNDING_ABSTAIN_MESSAGE = os.getenv(
    "LOW_GROUNDING_ABSTAIN_MESSAGE",
    "Not found in the regulations.",
)

# Per-regulation retrieval for multi-reg comparison queries
ENABLE_COMPARISON_RETRIEVAL = (
    os.getenv("ENABLE_COMPARISON_RETRIEVAL", "true").lower() == "true"
)
ENABLE_CLUSTER_RETRIEVAL = (
    os.getenv("ENABLE_CLUSTER_RETRIEVAL", "true").lower() == "true"
)
COMPARISON_CHUNKS_PER_REG = int(os.getenv("COMPARISON_CHUNKS_PER_REG", "3"))
CLUSTER_CHUNKS_PER_REG = int(os.getenv("CLUSTER_CHUNKS_PER_REG", "2"))

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "autosafety-rag")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

ENABLE_GATEWAY = os.getenv("ENABLE_GATEWAY", "false").lower() == "true"
EXPOSE_GATEWAY_API = os.getenv("EXPOSE_GATEWAY_API", "true").lower() == "true"

ENABLE_PROMETHEUS_METRICS = (
    os.getenv("ENABLE_PROMETHEUS_METRICS", "false").lower() == "true"
)

API_PREFIX = "/api/v1"
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080",
).split(",")
