"""Detect broad/summary queries that need wider retrieval and rerank pools."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_BROAD_TERMS = re.compile(
    r"\b("
    r"complete|comprehensive|full|entire|overall|summary|summarize|summarise|"
    r"overview|all\s+(?:of\s+)?(?:the\s+)?(?:requirements?|values?|numeric|provisions?)|"
    r"every\s+requirement|everything\s+that|list\s+all|enumerate|"
    r"all\s+numeric|all\s+applicable|across\s+both|multi-?attribute"
    r")\b",
    re.I,
)
_LIST_STYLE = re.compile(
    r"\b(?:including|covering|with)\s+(?:test\s+)?(?:loads?|angles?|durations?|"
    r"geometric|constraints?|requirements?)",
    re.I,
)
_MULTI_REQ = re.compile(
    r"\b(loads?|angles?|durations?|hold\s+times?|geometric|constraints?|"
    r"provisions?|requirements?|tests?)\b",
    re.I,
)


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


BROAD_RETRIEVAL_K = _i("BROAD_QUERY_RETRIEVAL_K", 20)
BROAD_RERANK_K = _i("BROAD_QUERY_RERANK_K", 18)
BROAD_FUSION_POOL_K = _i("BROAD_QUERY_FUSION_POOL_K", 28)


@dataclass(frozen=True)
class QueryBreadth:
    is_broad: bool
    signals: tuple[str, ...] = ()
    retrieval_k: int | None = None
    rerank_k: int | None = None
    fusion_pool_k: int | None = None

    @property
    def multi_attribute_summary(self) -> bool:
        return "multi_requirement_types" in self.signals or "list_style_attributes" in self.signals


def assess_query_breadth(query: str) -> QueryBreadth:
    """Return scaled k values when the query asks for a wide synthesis."""
    text = query or ""
    signals: list[str] = []

    if _BROAD_TERMS.search(text):
        signals.append("broad_summary_language")

    req_hits = {m.lower() for m in _MULTI_REQ.findall(text)}
    if len(req_hits) >= 3:
        signals.append("multi_requirement_types")

    if _LIST_STYLE.search(text):
        signals.append("list_style_attributes")

    is_broad = bool(signals) and (
        "broad_summary_language" in signals
        or len(req_hits) >= 3
        or ("list_style_attributes" in signals and len(req_hits) >= 2)
    )

    if not is_broad:
        return QueryBreadth(is_broad=False, signals=tuple(signals))

    return QueryBreadth(
        is_broad=True,
        signals=tuple(signals),
        retrieval_k=BROAD_RETRIEVAL_K,
        rerank_k=BROAD_RERANK_K,
        fusion_pool_k=BROAD_FUSION_POOL_K,
    )


def effective_retrieval_k(mode_k: int, breadth: QueryBreadth, *, default_pool: int) -> int:
    if breadth.is_broad and breadth.retrieval_k:
        return max(mode_k, breadth.retrieval_k)
    return mode_k or default_pool


def effective_rerank_k(breadth: QueryBreadth, *, default_k: int) -> int:
    if breadth.is_broad and breadth.rerank_k:
        return max(default_k, breadth.rerank_k)
    return default_k


def effective_fusion_pool_k(breadth: QueryBreadth, *, default_k: int) -> int:
    if breadth.is_broad and breadth.fusion_pool_k:
        return max(default_k, breadth.fusion_pool_k)
    return default_k
