from packages.analysis.comparability import assess_comparability
from packages.analysis.configuration import compare_configuration
from packages.analysis.global_response import compare_global_response
from packages.analysis.models import ComparabilityResult, ComparabilitySummary
from packages.analysis.quality import run_quality_gate
from packages.analysis.synthetic import generate_scenario


def _dim(summary: ComparabilitySummary, name: str) -> ComparabilityResult:
    return next(d for d in summary.dimensions if d.dimension == name)


def test_scn004_crash_pulse_blocks_causal_isolation() -> None:
    result = generate_scenario("SCN-004")
    quality_a = run_quality_gate(result.run_a.run_id, result.run_a.quality_raw)
    quality_b = run_quality_gate(result.run_b.run_id, result.run_b.quality_raw)
    global_response = compare_global_response(
        result.run_a.run_id,
        result.run_b.run_id,
        result.run_a.time_s,
        result.run_a.signals["vehicle_pulse"],
        result.run_b.signals["vehicle_pulse"],
    )
    config_diffs = compare_configuration(result.run_a.config, result.run_b.config)

    summary = assess_comparability(
        result.run_a.run_id,
        result.run_b.run_id,
        quality_a,
        quality_b,
        global_response=global_response,
        configuration_diffs=config_diffs,
    )
    assert _dim(summary, "global_pulse").status == "NOT_COMPARABLE"
    assert _dim(summary, "causal_isolation").status == "NOT_ESTABLISHED"
    assert summary.overall_status == "NOT_COMPARABLE"


def test_scn010_quality_failure_blocks_comparability() -> None:
    result = generate_scenario("SCN-010")
    quality_a = run_quality_gate(result.run_a.run_id, result.run_a.quality_raw)
    quality_b = run_quality_gate(result.run_b.run_id, result.run_b.quality_raw)

    summary = assess_comparability(result.run_a.run_id, result.run_b.run_id, quality_a, quality_b)
    assert _dim(summary, "occupant_response").status == "NOT_COMPARABLE"
    assert summary.overall_status == "NOT_COMPARABLE"


def test_scn008_processing_version_makes_primary_metric_conditional() -> None:
    result = generate_scenario("SCN-008")
    quality_a = run_quality_gate(result.run_a.run_id, result.run_a.quality_raw)
    quality_b = run_quality_gate(result.run_b.run_id, result.run_b.quality_raw)

    summary = assess_comparability(
        result.run_a.run_id,
        result.run_b.run_id,
        quality_a,
        quality_b,
        result_processing_version_a="CFC180",
        result_processing_version_b="CFC60",
    )
    assert _dim(summary, "primary_metric").status == "CONDITIONAL"


def test_scn001_fully_comparable_single_factor() -> None:
    result = generate_scenario("SCN-001")
    quality_a = run_quality_gate(result.run_a.run_id, result.run_a.quality_raw)
    quality_b = run_quality_gate(result.run_b.run_id, result.run_b.quality_raw)
    global_response = compare_global_response(
        result.run_a.run_id,
        result.run_b.run_id,
        result.run_a.time_s,
        result.run_a.signals["vehicle_pulse"],
        result.run_b.signals["vehicle_pulse"],
    )
    config_diffs = compare_configuration(result.run_a.config, result.run_b.config)

    summary = assess_comparability(
        result.run_a.run_id,
        result.run_b.run_id,
        quality_a,
        quality_b,
        global_response=global_response,
        configuration_diffs=config_diffs,
    )
    assert _dim(summary, "global_pulse").status == "COMPARABLE"
    assert _dim(summary, "causal_isolation").status == "COMPARABLE"
    assert summary.overall_status in ("COMPARABLE", "UNKNOWN")  # physical_correlation is UNKNOWN by default


def test_missing_inputs_are_not_established_not_guessed_pass() -> None:
    result = generate_scenario("SCN-001")
    quality_a = run_quality_gate(result.run_a.run_id, result.run_a.quality_raw)
    quality_b = run_quality_gate(result.run_b.run_id, result.run_b.quality_raw)
    summary = assess_comparability(result.run_a.run_id, result.run_b.run_id, quality_a, quality_b)
    assert _dim(summary, "global_pulse").status == "NOT_ESTABLISHED"
    assert _dim(summary, "causal_isolation").status == "NOT_ESTABLISHED"
