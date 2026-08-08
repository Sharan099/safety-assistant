"""Agent state, step traces, and final response models."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.citations import ClaimCheck
from generation.answer import SourceChunk

ToolName = Literal[
    "retrieve",
    "compare_regulations",
    "lookup_table",
    "draft_report",
    "design_implication",
    "applicability",
    "checklist_gen",
    "retest_scope",
    "plan",
    "synthesize",
]


class PlannedStep(BaseModel):
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class AgentStepTrace(BaseModel):
    step_id: str
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    output_preview: str = ""
    citations: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    citation_coverage: float = 1.0
    latency_ms: float = 0.0
    error: str | None = None
    started_at: float = Field(default_factory=time.time)


class AgentResult(BaseModel):
    """Final agent response + full step trace for evaluation."""

    task: str
    mode: str = "qa"  # qa | compare | report | design_implication | …
    answer: str = ""
    table_markdown: str = ""
    report_markdown: str = ""
    sources: list[SourceChunk] = Field(default_factory=list)
    plan: list[PlannedStep] = Field(default_factory=list)
    steps: list[AgentStepTrace] = Field(default_factory=list)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider: str = ""
    model: str = ""
    not_found: bool = False
    overall_citation_coverage: float = 1.0
    ungrounded_claim_count: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    query_intent: str | None = None
    execution_layer: str | None = None
    multi_step: bool = False
    mode_disclaimer: str | None = None
    mode_disclaimer_title: str | None = None

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)
