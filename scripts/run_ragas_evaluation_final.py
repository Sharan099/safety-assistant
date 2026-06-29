#!/usr/bin/env python3
"""
Full RAGAS evaluation pipeline → output/ragas_evaluation_final.csv + .png

Reuses run_ragas_eval.py (retrieval) and run_ragas_score.py (judge metrics).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "data" / "ragas_cases.json"
DEFAULT_CSV = ROOT / "output" / "ragas_evaluation_final.csv"
DEFAULT_PNG = ROOT / "output" / "ragas_evaluation_final.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--require-anthropic", action="store_true")
    args = parser.parse_args()

    retrieval_json = args.out_csv.with_suffix(".retrieval.json")
    py = sys.executable

    if not args.skip_retrieval:
        cmd = [
            py,
            str(ROOT / "scripts" / "run_ragas_eval.py"),
            "--cases",
            str(args.cases),
            "--out",
            str(args.out_csv),
            "--top-k",
            str(args.top_k),
            "--skip-ragas",
        ]
        print("=== Phase: live retrieval ===")
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    if not args.skip_scoring:
        score_cmd = [
            py,
            str(ROOT / "scripts" / "run_ragas_score.py"),
            str(retrieval_json),
            str(args.out_csv),
            "--resume",
        ]
        if args.require_anthropic:
            score_cmd.append("--require-anthropic")
        print("=== Phase: judge scoring (4 metrics) ===")
        subprocess.run(score_cmd, check=True, cwd=str(ROOT))

    if not args.skip_plot and args.out_csv.exists():
        plot_cmd = [py, str(ROOT / "scripts" / "ragas_plot_summary.py"), str(args.out_csv), str(args.out_png)]
        print("=== Phase: PNG + summary ===")
        subprocess.run(plot_cmd, check=True, cwd=str(ROOT))

    print(f"\nDone. CSV: {args.out_csv}")
    if args.out_png.exists():
        print(f"PNG:  {args.out_png}")


if __name__ == "__main__":
    main()
