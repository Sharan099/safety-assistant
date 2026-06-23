#!/usr/bin/env python3
"""Verify retrieval filtering — frontal vs side, legal vs rating."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CASES = [
    {
        "id": "frontal_no_side",
        "query": "What is the chest deflection legal limit in frontal impact under UN R94?",
        "forbidden_test_types": ["side", "pole_side", "rear"],
        "forbidden_doc_types": [],
    },
    {
        "id": "side_no_frontal",
        "query": "What side impact occupant protection requirements apply under UN R95?",
        "forbidden_test_types": ["frontal", "pole_side"],
        "forbidden_doc_types": [],
    },
    {
        "id": "legal_no_rating",
        "query": "What is the legal chest deflection requirement under UN R94?",
        "forbidden_test_types": [],
        "forbidden_doc_types": ["rating"],
        "forbidden_value_types": ["rating_threshold"],
    },
    {
        "id": "legal_no_reference",
        "query": "What is the official legal limit for belt anchorage strength under UN R14?",
        "forbidden_test_types": [],
        "forbidden_doc_types": ["reference"],
    },
]


def run_case(retriever, case: dict) -> dict:
    result = retriever.retrieve(case["query"])
    docs = result.get("documents", [])
    violations = []
    for d in docs:
        cid = d.get("id", "")
        chunk = retriever._chunk_by_id.get(cid, {})
        tt = chunk.get("test_type", "general")
        dt = chunk.get("doc_type", "")
        vt = chunk.get("value_type", "")
        if tt in case.get("forbidden_test_types", []):
            violations.append({"chunk_id": cid, "test_type": tt})
        if dt in case.get("forbidden_doc_types", []):
            violations.append({"chunk_id": cid, "doc_type": dt})
        if vt in case.get("forbidden_value_types", []):
            violations.append({"chunk_id": cid, "value_type": vt})
    return {
        "id": case["id"],
        "query": case["query"],
        "doc_count": len(docs),
        "violations": violations,
        "passed": len(violations) == 0,
    }


def main() -> int:
    from backend.app.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()
    results = [run_case(retriever, c) for c in CASES]
    out = ROOT / "output" / "verify_retrieval.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = sum(1 for r in results if r["passed"])
    print(f"Passed {passed}/{len(results)}")
    for r in results:
        status = "OK" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['doc_count']} docs, {len(r['violations'])} violations")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
