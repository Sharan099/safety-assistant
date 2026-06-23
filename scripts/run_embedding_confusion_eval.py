#!/usr/bin/env python3
"""Part D: embedding confusion set for §6.4 clause family (Nomic vs BGE-M3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFUSION_QUERIES = [
    {
        "id": "C01",
        "query": "UN R14 M1 front outboard anchorage load three-point retractor pulley",
        "expected_clause_prefix": "6.4.1",
    },
    {
        "id": "C02",
        "query": "UN R14 rear centre seat anchorage test without retractor geometry",
        "expected_clause_prefix": "6.4.2",
    },
    {
        "id": "C03",
        "query": "UN R14 lap belt lower anchorage test load",
        "expected_clause_prefix": "6.4.3",
    },
]


def _load_chunks() -> list[dict]:
    path = ROOT / "output" / "chunks.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("chunks", [])


def _top1_clause(chunks: list[dict], query: str) -> str | None:
    """Lexical proxy when embeddings unavailable (CI-safe)."""
    q = query.lower()
    scored: list[tuple[int, str]] = []
    for c in chunks:
        if c.get("regulation") != "UN_R14":
            continue
        clause = str(c.get("clause_number") or "")
        if not clause.startswith("6.4"):
            continue
        text = (c.get("text") or "").lower()
        score = sum(1 for w in q.split() if len(w) > 3 and w in text)
        if c.get("chunk_type") == "combined_context":
            score += 2
        scored.append((score, clause))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "output" / "embedding_confusion_eval.json"))
    args = parser.parse_args()

    chunks = _load_chunks()
    results = []
    correct = 0
    for item in CONFUSION_QUERIES:
        top = _top1_clause(chunks, item["query"])
        ok = bool(top and top.startswith(item["expected_clause_prefix"]))
        correct += int(ok)
        results.append({**item, "top1_clause": top, "correct": ok})

    report = {
        "mode": "lexical_proxy",
        "note": "Re-run with live dense retrieval after re-embed for Nomic vs BGE comparison.",
        "accuracy": correct / len(CONFUSION_QUERIES) if CONFUSION_QUERIES else 0,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if correct == len(CONFUSION_QUERIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
