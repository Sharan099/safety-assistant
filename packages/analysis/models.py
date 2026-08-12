"""Structured result types for the deterministic analysis layer.

TRD.md §23: "Tools return structured JSON/Pydantic objects." Every numerical
result carries `Provenance` (algorithm, version, parameters) per TRD.md §18 —
never a bare number with no way to trace how it was produced.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["PASS", "WARNING", "FAIL", "UNKNOWN"]
ComparabilityStatus = Literal["COMPARABLE", "CONDITIONAL", "NOT_COMPARABLE", "NOT_ESTABLISHED", "UNKNOWN"]
ChangeStatus = Literal["SAME", "CHANGED", "UNKNOWN"]
ChangeClassification = Literal["INTENTIONAL", "DEPENDENCY", "UNINTENTIONAL", "UNKNOWN"]


class Provenance(BaseModel):
    algorithm: str
    algorithm_version: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class QualityCheckResult(BaseModel):
    check_type: str
    status: Status
    value: dict[str, Any] | None = None
    threshold: dict[str, Any] | None = None
    explanation: str
    provenance: Provenance


class QualityGateSummary(BaseModel):
    run_id: str
    overall_status: Status
    checks: list[QualityCheckResult]


class ComparabilityResult(BaseModel):
    dimension: str
    status: ComparabilityStatus
    explanation: str
    evidence: list[str] = Field(default_factory=list)


class ComparabilitySummary(BaseModel):
    run_a_id: str
    run_b_id: str
    overall_status: ComparabilityStatus
    dimensions: list[ComparabilityResult]


class GlobalResponseMetric(BaseModel):
    name: str
    run_a_value: float | None
    run_b_value: float | None
    delta: float | None
    delta_pct: float | None
    materially_different: bool
    provenance: Provenance


class GlobalResponseComparison(BaseModel):
    run_a_id: str
    run_b_id: str
    metrics: list[GlobalResponseMetric]
    material_difference_detected: bool


class ConfigDiffEntry(BaseModel):
    path: str
    run_a_value: Any = None
    run_b_value: Any = None
    change_status: ChangeStatus
    change_classification: ChangeClassification


class SignalFeatureSet(BaseModel):
    signal: str
    run_id: str
    peak: float
    time_to_peak_ms: float
    rise_time_ms: float | None
    duration_ms: float | None
    integral: float
    provenance: Provenance


class DivergenceEvent(BaseModel):
    signal: str
    time_ms: float
    threshold: dict[str, Any]
    window: dict[str, Any]
    alignment_method: str
    source_run_a: str
    source_run_b: str
    provenance: Provenance


class SignalAnalysisResult(BaseModel):
    signal: str
    run_a_id: str
    run_b_id: str
    run_a_features: SignalFeatureSet
    run_b_features: SignalFeatureSet
    correlation: float
    divergence: DivergenceEvent | None
    provenance: Provenance
