"""assess_comparability() — PRD.md PR-004.

Comparability is evaluated per PR-004 dimension: global crash pulse,
occupant response, primary metric, physical-test correlation, causal
isolation. Quality and comparability stay separate functions (PRD.md §5:
"Quality and comparability are separate") — this one only *consumes*
`QualityGateSummary`/`GlobalResponseComparison`/`ConfigDiffEntry` results,
it never recomputes them.

Any dimension whose inputs haven't been computed yet is NOT_ESTABLISHED, not
guessed as COMPARABLE — an omission must never read as a pass.
"""

from __future__ import annotations

from packages.analysis.models import (
    ComparabilityResult,
    ComparabilityStatus,
    ComparabilitySummary,
    ConfigDiffEntry,
    GlobalResponseComparison,
    QualityGateSummary,
)

ALGORITHM_VERSION = "assess_comparability v0.1.0"

# Our own aggregation convention (PRD.md does not rank these): higher is
# "less trustworthy to treat as a straightforward comparison".
_SEVERITY: dict[ComparabilityStatus, int] = {
    "COMPARABLE": 0,
    "CONDITIONAL": 1,
    "UNKNOWN": 2,
    "NOT_ESTABLISHED": 3,
    "NOT_COMPARABLE": 4,
}


def _quality_based_status(
    quality_a: QualityGateSummary, quality_b: QualityGateSummary
) -> tuple[ComparabilityStatus, str]:
    statuses = {quality_a.overall_status, quality_b.overall_status}
    if "FAIL" in statuses:
        return "NOT_COMPARABLE", "At least one run fails its quality gate."
    if "WARNING" in statuses:
        return "CONDITIONAL", "At least one run has quality-gate warnings; treat conclusions with caution."
    if "UNKNOWN" in statuses:
        return "NOT_ESTABLISHED", "Quality status is not fully known for at least one run."
    return "COMPARABLE", "Both runs pass their quality gate."


def _global_pulse_dimension(global_response: GlobalResponseComparison | None) -> ComparabilityResult:
    if global_response is None:
        return ComparabilityResult(
            dimension="global_pulse",
            status="NOT_ESTABLISHED",
            explanation="Global crash response has not been evaluated yet.",
        )
    if global_response.material_difference_detected:
        diffs = [m.name for m in global_response.metrics if m.materially_different]
        return ComparabilityResult(
            dimension="global_pulse",
            status="NOT_COMPARABLE",
            explanation=f"Global crash response differs materially: {', '.join(diffs)}.",
            evidence=diffs,
        )
    return ComparabilityResult(
        dimension="global_pulse", status="COMPARABLE", explanation="Global crash response is comparable."
    )


def _causal_isolation_dimension(
    global_response: GlobalResponseComparison | None, configuration_diffs: list[ConfigDiffEntry] | None
) -> ComparabilityResult:
    if global_response is None or configuration_diffs is None:
        return ComparabilityResult(
            dimension="causal_isolation",
            status="NOT_ESTABLISHED",
            explanation="Global response and/or configuration diff have not been evaluated yet.",
        )
    if global_response.material_difference_detected:
        return ComparabilityResult(
            dimension="causal_isolation",
            status="NOT_ESTABLISHED",
            explanation="Global crash pulse differs materially; local-component causal isolation cannot be "
            "established until that is resolved (PR-005).",
        )
    changed_categories = {
        d.path.split(".")[0] for d in configuration_diffs if d.change_status in ("CHANGED", "UNKNOWN")
    }
    if not changed_categories:
        return ComparabilityResult(
            dimension="causal_isolation",
            status="COMPARABLE",
            explanation="No configuration category changed; any signal difference has no identified "
            "configuration-level mechanism.",
        )
    if len(changed_categories) == 1:
        return ComparabilityResult(
            dimension="causal_isolation",
            status="COMPARABLE",
            explanation=f"Exactly one configuration category changed ({next(iter(changed_categories))}); "
            "a single-factor comparison is possible.",
            evidence=sorted(changed_categories),
        )
    return ComparabilityResult(
        dimension="causal_isolation",
        status="CONDITIONAL",
        explanation=f"Multiple configuration categories changed ({', '.join(sorted(changed_categories))}); "
        "isolate before attributing causality to any single one.",
        evidence=sorted(changed_categories),
    )


def assess_comparability(
    run_a_id: str,
    run_b_id: str,
    quality_a: QualityGateSummary,
    quality_b: QualityGateSummary,
    *,
    global_response: GlobalResponseComparison | None = None,
    configuration_diffs: list[ConfigDiffEntry] | None = None,
    result_processing_version_a: str | None = None,
    result_processing_version_b: str | None = None,
    physical_test_available: bool = False,
) -> ComparabilitySummary:
    occupant_status, occupant_explanation = _quality_based_status(quality_a, quality_b)
    dimensions = [
        _global_pulse_dimension(global_response),
        ComparabilityResult(dimension="occupant_response", status=occupant_status, explanation=occupant_explanation),
    ]

    if (
        result_processing_version_a is not None
        and result_processing_version_b is not None
        and (result_processing_version_a != result_processing_version_b)
    ):
        dimensions.append(
            ComparabilityResult(
                dimension="primary_metric",
                status="CONDITIONAL",
                explanation=f"Result processing version differs ({result_processing_version_a} vs "
                f"{result_processing_version_b}); verify comparability before drawing conclusions from the "
                "primary metric.",
            )
        )
    else:
        metric_status, metric_explanation = _quality_based_status(quality_a, quality_b)
        dimensions.append(
            ComparabilityResult(dimension="primary_metric", status=metric_status, explanation=metric_explanation)
        )

    dimensions.append(
        ComparabilityResult(
            dimension="physical_correlation",
            status="COMPARABLE" if physical_test_available else "UNKNOWN",
            explanation="Physical test reference available."
            if physical_test_available
            else "No physical test reference provided for this investigation.",
        )
    )
    dimensions.append(_causal_isolation_dimension(global_response, configuration_diffs))

    overall = max((d.status for d in dimensions), key=lambda s: _SEVERITY[s])
    return ComparabilitySummary(run_a_id=run_a_id, run_b_id=run_b_id, overall_status=overall, dimensions=dimensions)
