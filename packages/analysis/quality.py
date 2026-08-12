"""run_quality_gate() — PRD.md PR-003, IMPLEMENTATION_PLAN.md Phase 7/8.

Reads a run's raw solver diagnostics (`SimulationRun.metadata_["quality_raw"]`
— see `scripts/generate_synthetic_dataset.py` for the synthetic shape, a real
`LSDYNAProvider` would populate the same shape from d3hsp/glstat/matsum) and
evaluates each check independently. A missing raw field produces UNKNOWN for
that check, never a silent PASS — PRD.md §5/PR-003: "Unknown is distinct from
pass."
"""

from __future__ import annotations

from typing import Any

from packages.analysis.models import Provenance, QualityCheckResult, QualityGateSummary, Status

ALGORITHM_VERSION = "run_quality_gate v0.1.0"

# (warning_threshold, fail_threshold) — exceeding warning -> WARNING, exceeding fail -> FAIL.
_BOUNDED_THRESHOLDS: dict[str, tuple[float, float]] = {
    "mass_scaling_pct": (3.0, 10.0),
    "energy_balance_pct": (5.0, 15.0),
    "hourglass_energy_pct": (5.0, 10.0),
    "contact_energy_pct": (8.0, 15.0),
    "penetration_max_mm": (1.0, 5.0),
}

_STATUS_RANK: dict[Status, int] = {"PASS": 0, "UNKNOWN": 1, "WARNING": 2, "FAIL": 3}


def _bounded_check(check_type: str, raw: dict[str, Any], field: str) -> QualityCheckResult:
    warn, fail = _BOUNDED_THRESHOLDS[field]
    value = raw.get(field)
    if value is None:
        return QualityCheckResult(
            check_type=check_type,
            status="UNKNOWN",
            explanation=f"'{field}' not present in run diagnostics.",
            provenance=Provenance(algorithm="run_quality_gate", algorithm_version=ALGORITHM_VERSION),
        )
    status: Status = "PASS" if value <= warn else ("WARNING" if value <= fail else "FAIL")
    return QualityCheckResult(
        check_type=check_type,
        status=status,
        value={field: value},
        threshold={"warning": warn, "fail": fail},
        explanation=f"{field}={value} (warning>{warn}, fail>{fail})",
        provenance=Provenance(
            algorithm="run_quality_gate",
            algorithm_version=ALGORITHM_VERSION,
            parameters={"field": field, "warning_threshold": warn, "fail_threshold": fail},
        ),
    )


def run_quality_gate(run_id: str, quality_raw: dict[str, Any] | None) -> QualityGateSummary:
    """Evaluate every PR-003 quality check for one run.

    `quality_raw` is `None` when the run has no diagnostics at all (every
    check becomes UNKNOWN, and overall status is UNKNOWN, not PASS).
    """
    raw = quality_raw or {}
    checks: list[QualityCheckResult] = []

    def prov(**params: Any) -> Provenance:
        return Provenance(algorithm="run_quality_gate", algorithm_version=ALGORITHM_VERSION, parameters=params)

    termination = raw.get("termination")
    if termination is None:
        checks.append(
            QualityCheckResult(
                check_type="solver_termination",
                status="UNKNOWN",
                explanation="No termination status recorded.",
                provenance=prov(),
            )
        )
    else:
        checks.append(
            QualityCheckResult(
                check_type="solver_termination",
                status="PASS" if termination == "NORMAL" else "FAIL",
                value={"termination": termination},
                explanation=f"Solver termination: {termination}.",
                provenance=prov(field="termination"),
            )
        )

    errors = raw.get("errors")
    warnings = raw.get("warnings")
    if errors is None and warnings is None:
        checks.append(
            QualityCheckResult(
                check_type="errors_warnings",
                status="UNKNOWN",
                explanation="No error/warning log recorded.",
                provenance=prov(),
            )
        )
    else:
        errors = errors or []
        warnings = warnings or []
        status: Status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        checks.append(
            QualityCheckResult(
                check_type="errors_warnings",
                status=status,
                value={"errors": errors, "warnings": warnings},
                explanation=f"{len(errors)} error(s), {len(warnings)} warning(s).",
                provenance=prov(field="errors/warnings"),
            )
        )

    final_time = raw.get("final_time_s")
    target_time = raw.get("target_time_s")
    if final_time is None or target_time is None:
        checks.append(
            QualityCheckResult(
                check_type="final_time",
                status="UNKNOWN",
                explanation="final_time_s or target_time_s missing.",
                provenance=prov(),
            )
        )
    else:
        reached = final_time >= target_time - 1e-9
        checks.append(
            QualityCheckResult(
                check_type="final_time",
                status="PASS" if reached else "FAIL",
                value={"final_time_s": final_time, "target_time_s": target_time},
                explanation=f"Reached {final_time}s of {target_time}s target."
                if reached
                else f"Terminated early at {final_time}s (target {target_time}s).",
                provenance=prov(field="final_time_s/target_time_s"),
            )
        )

    timestep_stable = raw.get("timestep_stable")
    if timestep_stable is None:
        checks.append(
            QualityCheckResult(
                check_type="timestep_history",
                status="UNKNOWN",
                explanation="timestep_stable not recorded.",
                provenance=prov(),
            )
        )
    else:
        checks.append(
            QualityCheckResult(
                check_type="timestep_history",
                status="PASS" if timestep_stable else "WARNING",
                value={"timestep_stable": timestep_stable},
                explanation="Timestep history stable." if timestep_stable else "Timestep history unstable.",
                provenance=prov(field="timestep_stable"),
            )
        )

    for check_type, field in [
        ("mass_scaling", "mass_scaling_pct"),
        ("energy_balance", "energy_balance_pct"),
        ("hourglass_energy", "hourglass_energy_pct"),
        ("contact_energy", "contact_energy_pct"),
        ("penetration", "penetration_max_mm"),
    ]:
        checks.append(_bounded_check(check_type, raw, field))

    failed_elements = raw.get("failed_elements")
    if failed_elements is None:
        checks.append(
            QualityCheckResult(
                check_type="failed_elements",
                status="UNKNOWN",
                explanation="failed_elements not recorded.",
                provenance=prov(),
            )
        )
    else:
        status = "PASS" if failed_elements == 0 else ("WARNING" if failed_elements <= 10 else "FAIL")
        checks.append(
            QualityCheckResult(
                check_type="failed_elements",
                status=status,
                value={"failed_elements": failed_elements},
                explanation=f"{failed_elements} failed/deleted element(s).",
                provenance=prov(field="failed_elements"),
            )
        )

    database_complete = raw.get("database_complete")
    if database_complete is None:
        checks.append(
            QualityCheckResult(
                check_type="result_database_completeness",
                status="UNKNOWN",
                explanation="database_complete not recorded.",
                provenance=prov(),
            )
        )
    else:
        checks.append(
            QualityCheckResult(
                check_type="result_database_completeness",
                status="PASS" if database_complete else "FAIL",
                value={"database_complete": database_complete},
                explanation="Result database complete." if database_complete else "Result database incomplete.",
                provenance=prov(field="database_complete"),
            )
        )

    overall = max((c.status for c in checks), key=lambda s: _STATUS_RANK[s], default="UNKNOWN")
    return QualityGateSummary(run_id=run_id, overall_status=overall, checks=checks)
