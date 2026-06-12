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

# Advanced retrieval stages
#   User Query -> Query Expansion -> Multi-Query -> Hybrid (Dense+BM25+RRF)
#   -> Metadata Filtering -> Parent-Child -> Cross-Encoder Reranker -> LLM
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
ENABLE_MULTI_QUERY = os.getenv("ENABLE_MULTI_QUERY", "true").lower() == "true"
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))
ENABLE_METADATA_FILTER = os.getenv("ENABLE_METADATA_FILTER", "true").lower() == "true"
METADATA_BOOST = float(os.getenv("METADATA_BOOST", "0.5"))
ENABLE_PARENT_CHILD = os.getenv("ENABLE_PARENT_CHILD", "true").lower() == "true"

# Grounding & anti-hallucination (Item 1)
#   If retrieval confidence is below threshold the bot abstains
#   ("not found in the corpus") instead of generating an answer.
ENABLE_GROUNDING_GATE = os.getenv("ENABLE_GROUNDING_GATE", "true").lower() == "true"
#   Min top semantic cosine similarity (0..1) to consider retrieval confident.
GROUNDING_MIN_SEMANTIC = float(os.getenv("GROUNDING_MIN_SEMANTIC", "0.45"))
#   Min cross-encoder rerank probability (sigmoid of logit) when reranker is on.
GROUNDING_MIN_RERANK_PROB = float(os.getenv("GROUNDING_MIN_RERANK_PROB", "0.5"))
#   Require inline [S#] citations in answers.
REQUIRE_CITATIONS = os.getenv("REQUIRE_CITATIONS", "true").lower() == "true"

ABSTAIN_MESSAGE = os.getenv(
    "ABSTAIN_MESSAGE",
    "I don't know — I could not find this in the indexed regulation corpus. "
    "Try rephrasing, naming a specific regulation (e.g. UN R14, UN R16), or "
    "upload the relevant document to this chat.",
)

# Observability
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "autosafety-rag")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

# Intelligent Multi-LLM Gateway (v3.0).
#   When disabled (default) the LangGraph workflow uses GroqLLM exactly as in
#   v2.2 — full backward compatibility. The detailed gateway knobs (providers,
#   pricing, routing weights, cache, reliability) live in
#   backend/app/gateway/config.py so the gateway stays a separable component.
ENABLE_GATEWAY = os.getenv("ENABLE_GATEWAY", "false").lower() == "true"
EXPOSE_GATEWAY_API = os.getenv("EXPOSE_GATEWAY_API", "true").lower() == "true"

# API
API_PREFIX = "/api/v1"
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080",
).split(",")
