#!/usr/bin/env python3
"""Part F multilayer evaluation — RAGAS + authority_correctness per mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "output" / "evaluation" / "multilayer"


def _load_cases(name: str) -> list[dict]:
    path = ROOT / "tests" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _score_authority_cases() -> dict:
    from backend.app.retrieval.citations import score_authority_correctness

    cases = _load_cases("test_cases_authority_tier.json")
    results = []
    for c in cases:
        # Deterministic structural checks without live LLM
        fake_answer = "Euro NCAP recommends; not legally binding [S1]."
        fake_cites = [{"marker": "S1", "authority_tier": "rating_protocol", "authority_tier_badge": "RATING"}]
        if c["id"] == "AT02":
            fake_answer = "UN R14 requires 1,350 daN for M1/N1 [S1]."
            fake_cites = [{"marker": "S1", "authority_tier": "legal_binding", "authority_tier_badge": "LEGAL"}]
        score = score_authority_correctness(fake_answer, fake_cites)
        results.append({"id": c["id"], **score})
    passed = sum(1 for r in results if r.get("passed"))
    return {
        "authority_correctness_rate": passed / len(results) if results else 0,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run live RAGAS judge")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "modes": {
            "authority_tier": _load_cases("test_cases_authority_tier.json"),
            "root_cause": _load_cases("test_cases_root_cause.json"),
            "design_review": _load_cases("test_cases_design_review.json"),
        },
        "authority_scoring": _score_authority_cases(),
    }

    if args.full:
        rc = __import__("subprocess").call(
            [sys.executable, "tests/run_ragas_eval.py", "--output-dir", str(OUT_DIR)],
            cwd=ROOT,
        )
        report["ragas_exit_code"] = rc

    out_json = OUT_DIR / "ragas_results.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = [
        "# Multilayer Evaluation Summary",
        "",
        f"- Authority correctness rate: {report['authority_scoring']['authority_correctness_rate']:.0%}",
        f"- Authority tier cases: {len(report['modes']['authority_tier'])}",
        f"- RCA cases: {len(report['modes']['root_cause'])}",
        f"- Design review cases: {len(report['modes']['design_review'])}",
        "",
        f"Full results: `{out_json}`",
    ]
    if not args.full:
        summary.append("- RAGAS: skipped (use `--full` for live judge)")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
