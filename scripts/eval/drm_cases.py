"""Per-case document-mismatch diff between two retrieval reports (baseline vs SAC): which
document each ranked first, and which cases flipped. Reads the JSON written by eval-retrieval /
sac_ab.py — no database, no LLM.

uv run python scripts/eval/drm_cases.py evals/results/retrieval_content_document_mismatch_v1_latest.json \
    evals/results/retrieval_sac_v1_document_mismatch_v1_latest.json --dataset evals/datasets/document_mismatch_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from safety_assistant.evaluation.dataset import load_dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--leg", default="full")
    ap.add_argument("--only", choices=["all", "flipped", "still_wrong"], default="flipped")
    args = ap.parse_args(argv)
    cases = {c.case_id: c for c in load_dataset(pathlib.Path(args.dataset)).cases}
    a = _leg(json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8")), args.leg)
    b = _leg(json.loads(pathlib.Path(args.candidate).read_text(encoding="utf-8")), args.leg)
    for cid, ca in a.items():
        cb = b.get(cid)
        case = cases.get(cid)
        if cb is None or case is None or not case.regulation_keys:
            continue
        x, y = ca["metrics"].get("drm@1"), cb["metrics"].get("drm@1")
        flipped = x != y
        if args.only == "flipped" and not flipped:
            continue
        if args.only == "still_wrong" and y != 1.0:
            continue
        verdict = "FIXED" if (x, y) == (1.0, 0.0) else "BROKEN" if (x, y) == (0.0, 1.0) else "same"
        neg = case.hard_negative_regulation_keys
        print(f"{cid}  [{verdict}]  expected {sorted(case.regulation_keys)}  hard-neg {neg}")
        print(f"  q: {case.query}")
        print(f"  baseline top: {ca['top_citations'][:1]}  docs {ca.get('ranked_documents', [])[:5]}")
        print(f"  candidate top: {cb['top_citations'][:1]}  docs {cb.get('ranked_documents', [])[:5]}")
    return 0


def _leg(report: dict, leg: str) -> dict[str, dict]:  # type: ignore[type-arg]
    lr = next(x for x in report["legs"] if x["leg"] == leg)
    return {c["case_id"]: c for c in lr["cases"]}


if __name__ == "__main__":
    sys.exit(main())
