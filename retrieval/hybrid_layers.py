"""Hybrid execution layers — fast deterministic vs bounded multi-step.

Layer 1–2 (fast): FACTUAL_LOOKUP, COMPLIANCE_CHECK — single-shot retrieval /
deterministic compliance. Also structured one-shot helpers (limits aggregation,
scope summary) stay off the agent loop.

Layer 3–5 (multi-step): DESIGN_IMPLICATION, APPLICABILITY, CHECKLIST_GEN,
RETEST_SCOPE — need concept expansion and/or per-regulation / per-category
loops that a single hybrid top-k call cannot do. These enter the bounded
multi-step / agent path only.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from retrieval.router import QueryIntent, RoutedQuery


class ExecutionLayer(str, Enum):
    FAST = "fast"  # Layer 1–2
    MULTI_STEP = "multi_step"  # Layer 3–5 bounded agent / intent pipelines


# Intents that must NOT enter the multi-step agent loop.
FAST_PATH_INTENTS: frozenset[QueryIntent] = frozenset(
    {
        QueryIntent.FACTUAL_LOOKUP,
        QueryIntent.COMPLIANCE_CHECK,
    }
)

# Intents that require the bounded multi-step layer (agent or intent pipelines).
MULTI_STEP_INTENTS: frozenset[QueryIntent] = frozenset(
    {
        QueryIntent.DESIGN_IMPLICATION,
        QueryIntent.APPLICABILITY,
        QueryIntent.CHECKLIST_GEN,
        QueryIntent.RETEST_SCOPE,
    }
)

# Structured one-shot (still chat/retrieve, not free-form agent tooling).
STRUCTURED_ONESHOT_INTENTS: frozenset[QueryIntent] = frozenset(
    {
        QueryIntent.SCOPE_SUMMARY,
    }
)


def execution_layer_for(intent: QueryIntent | RoutedQuery | str | None) -> ExecutionLayer:
    """Map a classified intent to the hybrid execution layer."""
    if isinstance(intent, RoutedQuery):
        qi = intent.intent
    elif isinstance(intent, QueryIntent):
        qi = intent
    elif isinstance(intent, str):
        try:
            qi = QueryIntent(intent)
        except ValueError:
            return ExecutionLayer.FAST
    else:
        return ExecutionLayer.FAST

    if qi in MULTI_STEP_INTENTS:
        return ExecutionLayer.MULTI_STEP
    return ExecutionLayer.FAST


def should_use_multi_step_layer(intent: QueryIntent | RoutedQuery | str | None) -> bool:
    return execution_layer_for(intent) is ExecutionLayer.MULTI_STEP


def is_fast_path_intent(intent: QueryIntent | RoutedQuery | str | None) -> bool:
    if isinstance(intent, RoutedQuery):
        qi = intent.intent
    elif isinstance(intent, QueryIntent):
        qi = intent
    elif isinstance(intent, str):
        try:
            qi = QueryIntent(intent)
        except ValueError:
            return True
    else:
        return True
    return qi in FAST_PATH_INTENTS or qi in STRUCTURED_ONESHOT_INTENTS


def layer_public_dict(routed: RoutedQuery | None) -> dict[str, Any]:
    if routed is None:
        return {
            "execution_layer": ExecutionLayer.FAST.value,
            "multi_step": False,
            "intent": None,
        }
    layer = execution_layer_for(routed)
    return {
        "execution_layer": layer.value,
        "multi_step": layer is ExecutionLayer.MULTI_STEP,
        "intent": routed.intent.value,
        "fast_path": routed.intent in FAST_PATH_INTENTS,
        "structured_oneshot": routed.intent in STRUCTURED_ONESHOT_INTENTS,
    }
