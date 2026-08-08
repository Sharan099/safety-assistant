"""Retrieval package — hybrid RRF + rewrite + rerank + small-to-big."""

__all__ = [
    "RetrievedChunk",
    "RewriteResult",
    "Reranker",
    "expand_to_parents",
    "format_context",
    "hybrid_search",
    "rerank",
    "retrieve",
    "rewrite_query",
    "rrf_merge",
]


def __getattr__(name: str):
    if name in {
        "RetrievedChunk",
        "format_context",
        "hybrid_search",
        "retrieve",
        "rrf_merge",
    }:
        from retrieval import retrieve as mod

        return getattr(mod, name)
    if name in {"RewriteResult", "rewrite_query"}:
        from retrieval import rewrite as mod

        return getattr(mod, name)
    if name in {"Reranker", "rerank"}:
        from retrieval import rerank as mod

        return getattr(mod, name)
    if name == "expand_to_parents":
        from retrieval.expand import expand_to_parents

        return expand_to_parents
    raise AttributeError(name)
