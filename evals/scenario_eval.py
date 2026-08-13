"""Deterministic-analysis evaluation — TRD.md Section 14/Phase 19 (Numerical):
feature accuracy, divergence accuracy, configuration-diff accuracy — run
against all 10 synthetic scenarios' ground truth (PRD.md Section 13).

Distinct from tests/analysis (which asserts specific known values): this
produces a scenario-by-scenario pass/fail report against each scenario's
`expected_signal_changes`, the way an engineer would audit the pipeline
before trusting it.

A signal "changed" here means detect_first_divergence found an event on the
index-aligned Run A/B pair — the same detector the product uses, not a
separate ad hoc check.

Usage:
    uv run python evals/scenario_eval.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.analysis.configuration import changed_paths, compare_configuration  # noqa: E402
from packages.analysis.quality import run_quality_gate  # noqa: E402
from packages.analysis.signals import detect_first_divergence  # noqa: E402
from packages.analysis.synthetic import DEFAULT_PARAMS, generate_all  # noqa: E402


def main() -> int:
    results = generate_all()
    all_ok = True

    print(f"{'scenario':<10} {'changed-signal recall':<24} {'quality_b':<10} {'ok':>4}")
    print("-" * 55)

    for result in results:
        spec, run_a, run_b = result.spec, result.run_a, result.run_b

        detected_changes = set()
        for signal_name in DEFAULT_PARAMS:
            event = detect_first_divergence(
                run_a.signals[signal_name],
                run_b.signals[signal_name],
                run_a.time_s,
                signal_name=signal_name,
                run_a_id=run_a.run_id,
                run_b_id=run_b.run_id,
            )
            if event is not None:
                detected_changes.add(signal_name)

        expected = set(spec.expected_signal_changes)
        missed = expected - detected_changes
        recall = 1.0 if not expected else len(expected & detected_changes) / len(expected)

        quality_b = run_quality_gate(run_b.run_id, run_b.quality_raw).overall_status
        # SCN-010's ground truth is specifically that quality gates the
        # investigation, not a signal-level claim.
        quality_ok = (quality_b == "FAIL") == (spec.scenario_id == "SCN-010")

        diffs = compare_configuration(run_a.config, run_b.config)
        config_changed = bool(changed_paths(diffs))
        # Every scenario except SCN-008 (processing-only) and SCN-009/SCN-010
        # (no intended config change) has some config diff.
        config_expected = spec.scenario_id not in ("SCN-008", "SCN-009", "SCN-010")
        config_ok = config_changed == config_expected

        ok = recall == 1.0 and quality_ok and config_ok
        all_ok = all_ok and ok

        print(
            f"{spec.scenario_id:<10} {recall:.2f} ({len(expected & detected_changes)}/{len(expected) or 0})".ljust(35)
            + f"{quality_b:<10} {'OK' if ok else 'FAIL':>4}"
        )
        if missed:
            print(f"           missed: {sorted(missed)}")
        if not quality_ok:
            print(f"           quality mismatch: got {quality_b}")
        if not config_ok:
            print(f"           config-diff mismatch: changed={config_changed}, expected={config_expected}")

    print("-" * 55)
    print("ALL SCENARIOS PASS" if all_ok else "SOME SCENARIOS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
