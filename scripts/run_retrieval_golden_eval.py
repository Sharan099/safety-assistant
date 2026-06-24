#!/usr/bin/env python3
"""Retrieval recall@k and regulation coverage on golden_retrieval_recall.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.document_registry import regulation_matches_corpus  # noqa: E402


def _chunk_reg(chunk: dict) -> str:
    return (chunk.get("regulation") or chunk.get("doc_id") or "").upper()


def _hit_regs(docs: list[dict], chunk_by_id: dict, expected: list[str]) -> set[str]:
    found: set[str] = set()
    for d in docs:
        c = chunk_by_id.get(d.get("id", ""), {})
        reg = _chunk_reg(c)
        for exp in expected:
            if regulation_matches_corpus(exp, reg):
                found.add(exp)
    return found


def _forbidden_present(docs: list[dict], chunk_by_id: dict, forbidden: list[str]) -> list[str]:
    bad: list[str] = []
    for d in docs:
        c = chunk_by_id.get(d.get("id", ""), {})
        reg = _chunk_reg(c)
        for f in forbidden:
            if regulation_matches_corpus(f, reg) and f not in bad:
                bad.append(f)
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=str(ROOT / "tests" / "golden_retrieval_recall.json"),
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output", default=str(ROOT / "output" / "golden_retrieval_eval.json"))
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    from backend.app.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()
    chunk_by_id = retriever._chunk_by_id

    results: list[dict] = []
    passed = 0
    for case in cases:
        mode = case.get("mode", "regulation_lookup")
        out = retriever.retrieve(case["question"], mode=mode)
        docs = out.get("documents", [])[: args.k]
        expected = case.get("expected_regs") or []
        forbidden = case.get("forbidden_regs") or []
        found = _hit_regs(docs, chunk_by_id, expected)
        bad = _forbidden_present(docs, chunk_by_id, forbidden)

        ok = True
        if expected and not set(expected).issubset(found):
            ok = False
        if bad:
            ok = False
        if not expected and docs and case.get("category") in ("ghost_reg",):
            ok = False
        if case.get("category") == "out_of_scope":
            ok = True

        if ok:
            passed += 1

        results.append({
            "id": case["id"],
            "category": case.get("category"),
            "passed": ok,
            "expected_regs": expected,
            "found_regs": sorted(found),
            "forbidden_hit": bad,
            "top_ids": [d.get("id") for d in docs[:5]],
            "retrieved": len(docs),
        })

    report = {
        "k": args.k,
        "total": len(cases),
        "passed": passed,
        "recall_rate": round(passed / len(cases), 4) if cases else 0,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Retrieval golden: {passed}/{len(cases)} passed (k={args.k})")
    print(f"Report: {out_path}")
    for r in results:
        if not r["passed"]:
            print(f"  FAIL {r['id']}: expected {r['expected_regs']} got {r['found_regs']} forbidden {r['forbidden_hit']}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
