"""Application settings — preserves original model choices."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OUTPUT_DIR,
    SYSTEM_PROMPT,
    TOP_K_CHUNKS,
    TOP_K_VECTOR,
    VECTOR_SCORE_THRESHOLD,
)

# Reranker (BGE cross-encoder). Note: bge-base-en-v1.5 is a bi-encoder
# embedding model; the matching reranker in the BGE family is bge-reranker-base.
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)

# Hybrid fusion
RRF_K = int(os.getenv("RRF_K", "60"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
TOP_K_AFTER_RERANK = int(os.getenv("TOP_K_AFTER_RERANK", "5"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "12"))

# Observability
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "autosafety-rag")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

# API
API_PREFIX = "/api/v1"
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080",
).split(",")
