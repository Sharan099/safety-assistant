"""Summary-augmented chunking: document summaries and retrieval representations.

Retrieval-only. Nothing in this package produces evidence text."""

from safety_assistant.contextualization.context_builder import (
    BASELINE_REPRESENTATION,
    SAC_COMPACT_REPRESENTATION,
    SAC_REPRESENTATION,
    ContextualizeStats,
    build_retrieval_text,
    compact_prefixes,
    contextualize_version,
    document_context,
)
from safety_assistant.contextualization.document_summary import ensure_summary, summary_cache_key
from safety_assistant.contextualization.prompts import SUMMARY_PROMPT_VERSION

__all__ = [
    "BASELINE_REPRESENTATION",
    "SAC_COMPACT_REPRESENTATION",
    "SAC_REPRESENTATION",
    "SUMMARY_PROMPT_VERSION",
    "ContextualizeStats",
    "build_retrieval_text",
    "compact_prefixes",
    "contextualize_version",
    "document_context",
    "ensure_summary",
    "summary_cache_key",
]
