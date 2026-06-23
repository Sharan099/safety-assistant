#!/usr/bin/env python3
"""Part G: final validation wrapper — deterministic + optional RAGAS full mode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "output" / "evaluation" / "final_validation"
CASES = ROOT / "tests" / "test_cases_final_validation.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Final validation harness (Part G)")
    parser.add_argument("--full", action="store_true", help="Run live RAGAS judge (slow)")
    parser.add_argument("--ragas-subset", type=int, default=5)
    args = parser.parse_args()

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    (FINAL_DIR / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")

    # Deterministic pytest gate
    rc = subprocess.call(
        [sys.executable, "-m", "pytest",
         "tests/test_clause_dependencies.py",
         "tests/test_applicability_grouping.py",
         "tests/test_table_structure_enrichment.py",
         "tests/test_corpus_uncertainty.py",
         "tests/test_gateway_tier_routing.py",
         "tests/test_extraction_fidelity.py",
         "-q"],
        cwd=ROOT,
    )
    if rc != 0:
        return rc

    # Embedding confusion eval (lexical proxy if no index)
    subprocess.call([sys.executable, "scripts/run_embedding_confusion_eval.py"], cwd=ROOT)

    summary_lines = [
        "# Final Validation Summary",
        "",
        f"- Cases: {len(cases)} (`tests/test_cases_final_validation.json`)",
        "- Deterministic regression: passed",
        "",
        "## Cases",
    ]
    for c in cases:
        summary_lines.append(f"- **{c['id']}** ({c['category']}): `{c['query'][:80]}…`")

    if args.full:
        ragas_rc = subprocess.call(
            [
                sys.executable,
                "tests/run_ragas_eval.py",
                "--ragas-subset",
                str(args.ragas_subset),
                "--output-dir",
                str(FINAL_DIR),
            ],
            cwd=ROOT,
        )
        summary_lines.append("")
        summary_lines.append(f"- RAGAS full mode: {'passed' if ragas_rc == 0 else 'failed'}")
        if ragas_rc != 0:
            (FINAL_DIR / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")
            return ragas_rc
    else:
        summary_lines.append("")
        summary_lines.append("- RAGAS: skipped (use `--full` for live judge)")

    (FINAL_DIR / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print((FINAL_DIR / "SUMMARY.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
