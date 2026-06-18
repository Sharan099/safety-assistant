#!/usr/bin/env python3
"""Move legacy evaluation artifacts into archive/ and promote v3.2 to current/."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "output" / "evaluation"
CURRENT = EVAL / "current"
ARCHIVE_V31 = EVAL / "archive" / "v3_1"
ARCHIVE_SNAP = EVAL / "archive" / "v3_2_snapshots"

V31_FILES = [
    "rag_eval_20_results.json",
    "eval_ragas_metrics.png",
    "eval_ragas_metrics_20.png",
    "eval_ablation_comparison.png",
    "eval_ablation_comparison_20.png",
    "eval_overall_scorecard.png",
    "eval_overall_scorecard_20.png",
    "eval_guardrails.png",
    "eval_guardrails_20.png",
    "eval_latency_distribution.png",
    "eval_latency_distribution_20.png",
]

ROOT_LEGACY = [
    ROOT / "output" / "rag_evaluation_results.json",
    ROOT / "output" / "rag_evaluation_comparison.png",
]


def _move(src: Path, dest_dir: Path) -> None:
    if not src.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    print(f"archived: {src.name} -> {dest_dir.relative_to(ROOT)}")


def main() -> None:
    CURRENT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_V31.mkdir(parents=True, exist_ok=True)
    ARCHIVE_SNAP.mkdir(parents=True, exist_ok=True)

    for name in V31_FILES:
        _move(EVAL / name, ARCHIVE_V31)

    for path in ROOT_LEGACY:
        _move(path, ARCHIVE_V31)

    v32_json = EVAL / "rag_eval_20_results_v3_2.json"
    if v32_json.exists():
        dest = CURRENT / "rag_eval_20_results.json"
        shutil.copy2(v32_json, dest)
        _move(v32_json, ARCHIVE_SNAP)
        print(f"promoted: rag_eval_20_results.json -> evaluation/current/")

    for png in EVAL.glob("eval_*.png"):
        _move(png, CURRENT)

    print("Done. Active eval: output/evaluation/current/")


if __name__ == "__main__":
    main()
