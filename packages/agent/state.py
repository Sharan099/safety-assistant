"""InvestigationState — TRD.md Section 24.

A `TypedDict` (LangGraph's native state shape) rather than a Pydantic model:
nodes read/write partial updates and LangGraph merges them into the running
state between steps.
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict


class InvestigationState(TypedDict, total=False):
    investigation_id: uuid.UUID
    question: str
    run_a_id: str
    run_b_id: str
    primary_metric: str

    # Each of these is a `.model_dump()` of the matching packages.analysis.models
    # type (QualityGateSummary / ComparabilitySummary / GlobalResponseComparison)
    # — plain dicts, not the Pydantic objects, so the whole state stays
    # JSON-serializable (this is what a LangGraph checkpointer would persist).
    quality_a: dict[str, Any]
    quality_b: dict[str, Any]
    comparability: dict[str, Any]
    global_response: dict[str, Any]
    configuration_diff: list[dict[str, Any]]

    signal_plan: list[str]
    signal_results: dict[str, dict[str, Any]]  # signal name -> SignalAnalysisResult.model_dump()

    knowledge_evidence: list[dict[str, Any]]
    historical_cases: list[dict[str, Any]]

    hypotheses: list[dict[str, Any]]
    unknowns: list[str]

    blocked_reason: str | None
    review_required: bool
