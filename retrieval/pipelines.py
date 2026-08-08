"""Intent → retrieval strategy helpers (used by ``retrieve``).

Each ``QueryIntent`` has its own ``PipelineConfig`` in ``retrieval.router``.
This module maps those configs onto retrieve-time overrides without collapsing
strategies into one shared knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retrieval.router import (
    PipelineConfig,
    QueryIntent,
    RetrievalStrategy,
    RoutedQuery,
    pipeline_for,
)


@dataclass
class RetrieveOverrides:
    """Concrete retrieve knobs derived from a ``PipelineConfig``."""

    hybrid_top_k: int
    rerank_top_k: int
    hard_reg_filter: bool
    force_multi_reg: bool
    force_enumerative: bool
    prepend_scope: bool
    small_to_big: bool
    strategy: RetrievalStrategy
    budget_mode: str
    intent: QueryIntent


def overrides_from_routed(routed: RoutedQuery | None) -> RetrieveOverrides | None:
    if routed is None:
        return None
    return overrides_from_pipeline(routed.pipeline)


def overrides_from_pipeline(pipe: PipelineConfig) -> RetrieveOverrides:
    strategy = pipe.retrieval_strategy
    force_multi = pipe.multi_reg_loop or strategy in {
        RetrievalStrategy.MULTI_REG_TOPIC,
        RetrievalStrategy.APPLICABILITY_SCOPES,
    }
    force_enum = strategy == RetrievalStrategy.CHECKLIST
    prepend = pipe.prepend_scope or strategy == RetrievalStrategy.SCOPE_ARTICLE
    return RetrieveOverrides(
        hybrid_top_k=pipe.hybrid_top_k,
        rerank_top_k=pipe.rerank_top_k,
        hard_reg_filter=pipe.hard_reg_filter,
        force_multi_reg=force_multi,
        force_enumerative=force_enum,
        prepend_scope=prepend,
        small_to_big=pipe.small_to_big,
        strategy=strategy,
        budget_mode=pipe.budget_mode,
        intent=pipe.intent,
    )


def intent_prompt_extra(routed: RoutedQuery | None) -> str:
    """User-prompt appendix from the routed pipeline's grounding instruction."""
    if routed is None:
        return ""
    text = (routed.pipeline.prompt_instruction or "").strip()
    return f"{text}\n\n" if text else ""


def default_pipeline(intent: QueryIntent) -> PipelineConfig:
    return pipeline_for(intent)


def record_intent_on_trace(routed: RoutedQuery | None, *, trace: Any | None = None) -> None:
    if routed is None:
        return
    try:
        if trace is None:
            from observability.context import get_current_trace

            trace = get_current_trace()
        if trace is None:
            return
        trace.optimizations["query_intent"] = routed.to_public_dict()
    except Exception:  # noqa: BLE001
        pass
