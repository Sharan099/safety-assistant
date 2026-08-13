"""Synthetic CAE benchmark — PRD.md §13, IMPLEMENTATION_PLAN.md Phase 5.

Ten scenarios (SCN-001..SCN-010), each a deterministic Run A / Run B pair
with a known changed factor, expected signal changes, and explicitly
allowed/disallowed conclusions. Every array is generated from a fixed
`numpy.random.default_rng` seed — same scenario in, byte-identical signals
out, always.

Physics here is a stand-in (Gaussian pulses), not a solver: it exists to
give `packages/analysis` something with known ground truth to be tested
against, per PRD.md §13's "known changed factor / expected signal changes /
allowed conclusions / disallowed conclusions" contract. Real solver data
(TRD.md §21, LSDYNAAdapter) replaces this later without changing that
contract's shape.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SAMPLE_RATE_HZ = 5000.0
DT_S = 1.0 / SAMPLE_RATE_HZ
DURATION_S = 0.150
TIME_S = np.round(np.arange(0.0, DURATION_S + DT_S, DT_S), 6)  # 751 samples, 0..150 ms

# Baseline pulse parameters per canonical signal name. `unit` and `fire_time`
# are metadata; everything else feeds the Gaussian pulse.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "vehicle_pulse": {"peak": 28.0, "t_peak": 0.025, "width": 0.009, "unit": "g"},
    "chest_acceleration": {"peak": 45.0, "t_peak": 0.045, "width": 0.012, "unit": "g"},
    "chest_deflection": {"peak": 35.0, "t_peak": 0.055, "width": 0.016, "unit": "mm"},
    "chest_velocity": {"peak": 7.5, "t_peak": 0.040, "width": 0.013, "unit": "m/s"},
    "belt_force": {"peak": 4000.0, "t_peak": 0.035, "width": 0.010, "unit": "N"},
    "pelvis_acceleration": {"peak": 38.0, "t_peak": 0.040, "width": 0.011, "unit": "g"},
    "torso_rotation": {"peak": 25.0, "t_peak": 0.050, "width": 0.014, "unit": "deg"},
    "airbag_pressure": {"peak": 120.0, "t_peak": 0.032, "width": 0.010, "unit": "kPa", "fire_time": 0.018},
}

DEFAULT_CONFIG: dict[str, Any] = {
    "seat": {"fore_aft_position_mm": 0, "recline_deg": 23},
    "restraint": {
        "belt": {"routing": "STANDARD", "webbing_revision": "B-17"},
        "pretensioner": {"fire_time_ms": 8.0},
        "force_limiter": {"level_n": 4000},
    },
    "airbag": {"fire_time_ms": 18.0, "vent_config": "STD"},
    "dummy": {"type": "THOR-50th", "posture": "NOMINAL"},
    "contacts": {"belt_torso_friction": 0.30},
    "solver": {"mass_scaling_pct": 0.4},
}

DEFAULT_QUALITY_RAW: dict[str, Any] = {
    "termination": "NORMAL",
    "errors": [],
    "warnings": [],
    "final_time_s": DURATION_S,
    "target_time_s": DURATION_S,
    "timestep_stable": True,
    "mass_scaling_pct": 0.4,
    "energy_balance_pct": 1.2,
    "hourglass_energy_pct": 3.5,
    "contact_energy_pct": 2.0,
    "penetration_max_mm": 0.05,
    "failed_elements": 0,
    "database_complete": True,
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _pulse(t: np.ndarray, peak: float, t_peak: float, width: float) -> np.ndarray:
    return peak * np.exp(-0.5 * ((t - t_peak) / width) ** 2)


def _generate_signals(
    params: dict[str, dict[str, Any]], *, seed: int, noise_frac: float = 0.01
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    signals: dict[str, np.ndarray] = {}
    for name, p in params.items():
        base = _pulse(TIME_S, p["peak"], p["t_peak"], p["width"])
        if "fire_time" in p:
            base = np.where(TIME_S < p["fire_time"], 0.0, base)
        noise = rng.normal(0.0, abs(p["peak"]) * noise_frac, size=TIME_S.shape)
        signals[name] = base + noise
    return signals


@dataclass
class RunFixture:
    run_id: str
    label: str  # "A" | "B"
    model_version: str
    config: dict[str, Any]
    quality_raw: dict[str, Any]
    signals: dict[str, np.ndarray]
    time_s: np.ndarray = field(default_factory=lambda: TIME_S)


@dataclass
class ScenarioSpec:
    scenario_id: str
    title: str
    changed_factor: str
    description: str
    expected_signal_changes: list[str]
    allowed_conclusions: list[str]
    disallowed_conclusions: list[str]


@dataclass
class ScenarioResult:
    spec: ScenarioSpec
    run_a: RunFixture
    run_b: RunFixture


def _build(
    *,
    index: int,
    scenario_id: str,
    title: str,
    changed_factor: str,
    description: str,
    expected_signal_changes: list[str],
    allowed_conclusions: list[str],
    disallowed_conclusions: list[str],
    config_overrides_b: dict[str, Any] | None = None,
    params_overrides_b: dict[str, Any] | None = None,
    quality_overrides_b: dict[str, Any] | None = None,
    model_version_b: str = "v12.3",
    param_scale_b: float | None = None,
    jitter_seed_b: int | None = None,
) -> ScenarioResult:
    run_a_params = copy.deepcopy(DEFAULT_PARAMS)
    run_b_params = _deep_merge(DEFAULT_PARAMS, params_overrides_b or {})

    if param_scale_b is not None:
        for p in run_b_params.values():
            p["peak"] *= param_scale_b
            p["width"] *= 1.0 / param_scale_b  # heavier filtering: lower peak, wider pulse

    if jitter_seed_b is not None:
        rng = np.random.default_rng(jitter_seed_b)
        for p in run_b_params.values():
            p["peak"] *= float(1.0 + rng.normal(0.0, 0.02))
            p["t_peak"] += float(rng.normal(0.0, 0.001))

    run_a = RunFixture(
        run_id=f"{scenario_id}-RUN-A",
        label="A",
        model_version="v12.3",
        config=copy.deepcopy(DEFAULT_CONFIG),
        quality_raw=copy.deepcopy(DEFAULT_QUALITY_RAW),
        signals=_generate_signals(run_a_params, seed=1000 + index),
    )
    run_b = RunFixture(
        run_id=f"{scenario_id}-RUN-B",
        label="B",
        model_version=model_version_b,
        config=_deep_merge(DEFAULT_CONFIG, config_overrides_b or {}),
        quality_raw=_deep_merge(DEFAULT_QUALITY_RAW, quality_overrides_b or {}),
        signals=_generate_signals(run_b_params, seed=2000 + index),
    )
    spec = ScenarioSpec(
        scenario_id=scenario_id,
        title=title,
        changed_factor=changed_factor,
        description=description,
        expected_signal_changes=expected_signal_changes,
        allowed_conclusions=allowed_conclusions,
        disallowed_conclusions=disallowed_conclusions,
    )
    return ScenarioResult(spec=spec, run_a=run_a, run_b=run_b)


def _scn_001() -> ScenarioResult:
    return _build(
        index=1,
        scenario_id="SCN-001",
        title="Belt revision",
        changed_factor="restraint.belt.force_limiter",
        description="Belt force-limiter level reduced from 4000 N to 3500 N between Run A and Run B.",
        expected_signal_changes=["belt_force", "chest_deflection"],
        allowed_conclusions=[
            "Belt force-limiter level differs (4000 N -> 3500 N); this is the only restraint-affecting "
            "configuration change and is consistent with the increased chest deflection.",
            "Runs are COMPARABLE on global crash pulse, occupant response, and primary metric.",
        ],
        disallowed_conclusions=[
            "The chest deflection increase is caused by a model revision (model_version is identical).",
            "Airbag deployment timing changed (it did not).",
        ],
        config_overrides_b={"restraint": {"force_limiter": {"level_n": 3500}, "belt": {"webbing_revision": "B-18"}}},
        params_overrides_b={"belt_force": {"peak": 3500.0}, "chest_deflection": {"peak": 40.0}},
    )


def _scn_002() -> ScenarioResult:
    return _build(
        index=2,
        scenario_id="SCN-002",
        title="Pretensioner timing",
        changed_factor="restraint.pretensioner.fire_time",
        description="Pretensioner fire time shifted from 8 ms to 14 ms.",
        expected_signal_changes=["belt_force"],
        allowed_conclusions=[
            "Pretensioner fire time shifted (8 ms -> 14 ms); belt force onset/peak timing shifted "
            "correspondingly, consistent with later restraint engagement.",
            "Chest deflection peak magnitude is materially unchanged; the effect is primarily temporal.",
        ],
        disallowed_conclusions=[
            "The belt force-limiter level changed (it did not).",
            "Airbag deployment timing changed (it did not).",
        ],
        config_overrides_b={"restraint": {"pretensioner": {"fire_time_ms": 14.0}}},
        params_overrides_b={"belt_force": {"t_peak": 0.041}},
    )


def _scn_003() -> ScenarioResult:
    return _build(
        index=3,
        scenario_id="SCN-003",
        title="Airbag deployment timing",
        changed_factor="airbag.fire_time",
        description="Airbag fire time delayed from 18 ms to 26 ms.",
        expected_signal_changes=["airbag_pressure", "chest_acceleration", "chest_deflection"],
        allowed_conclusions=[
            "Airbag fire time delayed (18 ms -> 26 ms); the occupant loads the restraint system longer "
            "before airbag engagement, consistent with increased chest acceleration and deflection.",
            "Belt and seat configuration are unchanged between runs.",
        ],
        disallowed_conclusions=[
            "Seat position changed (it did not).",
            "A pretensioner timing change explains the difference (it did not change).",
        ],
        config_overrides_b={"airbag": {"fire_time_ms": 26.0}},
        params_overrides_b={
            "airbag_pressure": {"fire_time": 0.026, "t_peak": 0.040},
            "chest_acceleration": {"peak": 52.0},
            "chest_deflection": {"peak": 39.0},
        },
    )


def _scn_004() -> ScenarioResult:
    return _build(
        index=4,
        scenario_id="SCN-004",
        title="Crash pulse change",
        changed_factor="vehicle.global_crash_pulse",
        description="Global vehicle crash pulse changed (structure/barrier revision), cascading downstream.",
        expected_signal_changes=[
            "vehicle_pulse",
            "chest_acceleration",
            "chest_deflection",
            "pelvis_acceleration",
        ],
        # belt_force does shift too (+7.5% peak) but stays under
        # detect_first_divergence's 10% threshold — a real detector
        # sensitivity limit, not a generator bug (evals/scenario_eval.py
        # caught this; documented rather than tuned away).
        allowed_conclusions=[
            "Global vehicle crash pulse differs materially (peak and timing) between Run A and Run B; "
            "per PR-005 this must be resolved before attributing downstream occupant-response differences "
            "to restraint/local components.",
            "Comparability for causal isolation of a local (restraint) mechanism is NOT_ESTABLISHED until "
            "the global pulse difference is explained.",
        ],
        disallowed_conclusions=[
            "The restraint system (belt/pretensioner/airbag) is the primary cause of the occupant-response "
            "difference — that configuration is identical between runs.",
            "Runs are COMPARABLE for causal isolation of a local component.",
        ],
        config_overrides_b={"solver": {"note": "structure/barrier revision affecting global pulse"}},
        params_overrides_b={
            "vehicle_pulse": {"peak": 34.0, "t_peak": 0.021, "width": 0.008},
            "chest_acceleration": {"peak": 55.0, "t_peak": 0.040},
            "chest_deflection": {"peak": 41.0},
            "belt_force": {"peak": 4300.0},
            "pelvis_acceleration": {"peak": 44.0},
        },
    )


def _scn_005() -> ScenarioResult:
    return _build(
        index=5,
        scenario_id="SCN-005",
        title="Seat position",
        changed_factor="seat.fore_aft_position",
        description="Seat fore-aft position moved 25 mm forward.",
        expected_signal_changes=["torso_rotation", "chest_deflection", "belt_force"],
        allowed_conclusions=[
            "Seat fore-aft position moved 25 mm forward in Run B; timing shifts in torso rotation, chest "
            "deflection and belt force onset are consistent with changed initial occupant/belt geometry.",
            "Restraint hardware configuration (belt, pretensioner, force limiter, airbag) is identical between runs.",
        ],
        disallowed_conclusions=[
            "Solver version changed (it did not).",
            "A belt hardware change explains the difference (belt configuration is identical).",
        ],
        config_overrides_b={"seat": {"fore_aft_position_mm": 25}},
        params_overrides_b={
            "chest_deflection": {"t_peak": 0.050},
            "torso_rotation": {"t_peak": 0.045, "peak": 28.0},
            "belt_force": {"t_peak": 0.032},
        },
    )


def _scn_006() -> ScenarioResult:
    return _build(
        index=6,
        scenario_id="SCN-006",
        title="Dummy positioning",
        changed_factor="dummy.posture",
        description="Dummy initial posture differs: pelvis rotated 5 degrees.",
        expected_signal_changes=["pelvis_acceleration", "torso_rotation"],
        allowed_conclusions=[
            "Dummy initial posture differs (pelvis rotated 5 degrees) between runs; pelvis acceleration and "
            "torso rotation differ in timing/magnitude consistent with this positioning change.",
            "Belt, pretensioner, airbag and seat configuration are identical between runs.",
        ],
        disallowed_conclusions=[
            "A restraint-system change explains the difference (none is present).",
            "Chest deflection changed for a restraint-related reason without inspecting the chest-deflection "
            "signal analysis first.",
        ],
        config_overrides_b={"dummy": {"posture": "PELVIS_ROTATED_5DEG"}},
        params_overrides_b={
            "pelvis_acceleration": {"peak": 46.0, "t_peak": 0.037},
            "torso_rotation": {"peak": 31.0, "t_peak": 0.046},
        },
    )


def _scn_007() -> ScenarioResult:
    return _build(
        index=7,
        scenario_id="SCN-007",
        title="Contact/friction change",
        changed_factor="contacts.belt_torso_friction",
        description="Belt-to-torso friction coefficient increased from 0.30 to 0.55.",
        # chest_deflection does shift too (35.0 -> 36.5, +4.3%) but stays
        # under the 10% divergence threshold — real detector sensitivity
        # limit, documented rather than tuned away (evals/scenario_eval.py).
        expected_signal_changes=["belt_force"],
        allowed_conclusions=[
            "Belt-to-torso friction coefficient increased (0.30 -> 0.55); belt force pulse shape and chest "
            "deflection differ modestly, consistent with a contact/friction change.",
            "Run B's contact-energy quality check is WARNING; treat local-signal conclusions with "
            "corresponding caution per PR-003.",
        ],
        disallowed_conclusions=[
            "The belt force-limiter level changed (it did not).",
            "This is a model revision without an intended physical change (a specific friction parameter "
            "change is identified in the configuration diff).",
        ],
        config_overrides_b={"contacts": {"belt_torso_friction": 0.55}},
        params_overrides_b={"belt_force": {"width": 0.013}, "chest_deflection": {"peak": 36.5}},
        quality_overrides_b={"contact_energy_pct": 9.0, "hourglass_energy_pct": 4.0},
    )


def _scn_008() -> ScenarioResult:
    return _build(
        index=8,
        scenario_id="SCN-008",
        title="Signal-processing change",
        changed_factor="result_processing_version",
        description="Result processing filter class differs (Run A: CFC180, Run B: CFC60); no hardware change.",
        expected_signal_changes=[
            "vehicle_pulse",
            "chest_acceleration",
            "chest_deflection",
            "belt_force",
            "pelvis_acceleration",
            "torso_rotation",
        ],
        allowed_conclusions=[
            "result_processing_version differs (CFC180 vs CFC60) while every hardware/component "
            "configuration entry is identical between Run A and Run B.",
            "Apparent differences across nearly all signals are consistent with a signal-processing (filter "
            "class) change rather than a physical change; primary-metric comparability should be marked "
            "CONDITIONAL/NOT_ESTABLISHED pending re-processing with a consistent filter.",
        ],
        disallowed_conclusions=[
            "Any specific restraint or seat hardware change explains the difference (none exists in configuration).",
            "The observed chest deflection difference is evidence of a physical belt or airbag change.",
        ],
        config_overrides_b={},
        # A 25% peak attenuation is realistic for CFC180 -> CFC60 (CFC60 is a
        # substantially more aggressive low-pass filter) and reliably clears
        # detect_first_divergence's 10% threshold on every signal, matching
        # the "nearly all signals differ" claim below (evals/scenario_eval.py
        # caught 0.90 being too conservative to actually demonstrate that).
        param_scale_b=0.75,
    )


def _scn_009() -> ScenarioResult:
    return _build(
        index=9,
        scenario_id="SCN-009",
        title="Model revision without intended physical change",
        changed_factor="model_version",
        description="Model version bumped (mesh/formulation refinement) with no restraint/seat/airbag/"
        "dummy/contact configuration change recorded.",
        # Jitter is deliberately small (~2%) — by design, no signal should
        # reliably cross detect_first_divergence's 10% threshold. An empty
        # list here is the honest ground truth, not a placeholder: it's
        # exactly what "differences are small" (below) should mean measured
        # against the product's own detector, not just prose.
        expected_signal_changes=[],
        allowed_conclusions=[
            "Model revision changed (v12.3 -> v12.4) with no corresponding restraint/seat/airbag/dummy/"
            "contact configuration change recorded; observed signal differences are small and consistent "
            "with expected numerical variation from a mesh/formulation revision.",
            "No specific causal mechanism is established; classification of the model-version change is "
            "UNKNOWN pending an engineering changelog.",
        ],
        disallowed_conclusions=[
            "The belt force-limiter or pretensioner timing changed (neither did).",
            "A definitive physical mechanism is identified for the small signal differences.",
        ],
        config_overrides_b={},
        model_version_b="v12.4",
        jitter_seed_b=42,
    )


def _scn_010() -> ScenarioResult:
    return _build(
        index=10,
        scenario_id="SCN-010",
        title="Numerical-quality failure",
        changed_factor="solver.numerical_quality",
        description="Run B terminates with an error before reaching target simulation time.",
        expected_signal_changes=[],
        allowed_conclusions=[
            "Run B fails the quality gate: solver terminated with an error (negative volume) before "
            "reaching target simulation time (81.2 ms vs 150 ms target); the result database is incomplete.",
            "No comparability or hypothesis conclusion should be drawn from Run B's signals until the "
            "numerical-quality failure is resolved.",
        ],
        disallowed_conclusions=[
            "Run A and Run B are COMPARABLE.",
            "Any specific restraint/seat/airbag mechanism explains a signal difference (the quality failure "
            "precludes a reliable comparison).",
        ],
        config_overrides_b={},
        quality_overrides_b={
            "hourglass_energy_pct": 12.0,
            "termination": "ERROR",
            "errors": ["negative volume element 48213 at t=0.0812s"],
            "final_time_s": 0.0812,
            "database_complete": False,
        },
    )


_BUILDERS = [_scn_001, _scn_002, _scn_003, _scn_004, _scn_005, _scn_006, _scn_007, _scn_008, _scn_009, _scn_010]


def generate_scenario(scenario_id: str) -> ScenarioResult:
    for builder in _BUILDERS:
        result = builder()
        if result.spec.scenario_id == scenario_id:
            return result
    raise ValueError(f"unknown scenario_id: {scenario_id}")


def generate_all() -> list[ScenarioResult]:
    return [builder() for builder in _BUILDERS]
