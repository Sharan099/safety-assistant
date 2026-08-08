"""Diagnose what outranks the correct clause for near-miss compliance cases."""

from __future__ import annotations

import json
from pathlib import Path

from retrieval.retrieve import hybrid_search
from retrieval.value_limit import bias_chunks_for_value_vs_limit, is_value_vs_limit_query

ROOT = Path(__file__).resolve().parents[1]
cases = {
    json.loads(l)["id"]: json.loads(l)
    for l in (ROOT / "eval/golden_set.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
}

for cid in ("cmp_003", "cmp_004", "num_005", "cmp_001", "num_003", "num_004"):
    c = cases[cid]
    q = c["question"]
    gold = set(c.get("expected_chunk_ids") or [])
    reg = c.get("regulation_scope") or c.get("regulation_id")
    print("\n" + "=" * 80)
    print(cid, "vvL=", is_value_vs_limit_query(q))
    print("Q:", q)
    print("gold:", gold)
    hits = hybrid_search(q, regulation_id=reg, top_k=30)
    biased = bias_chunks_for_value_vs_limit(hits, question=q)
    print("--- hybrid top5 ---")
    for i, ch in enumerate(hits[:5], 1):
        mark = "*" if ch.chunk_id in gold else " "
        preview = (ch.text or "").replace("\n", " ")[:110]
        print(f"{mark}{i} {ch.chunk_id} sec={ch.section_number!r} score={ch.score:.4f}")
        print(f"   {preview!r}")
    print("--- after value_vs_limit bias top5 ---")
    for i, ch in enumerate(biased[:5], 1):
        mark = "*" if ch.chunk_id in gold else " "
        preview = (ch.text or "").replace("\n", " ")[:110]
        print(f"{mark}{i} {ch.chunk_id} sec={ch.section_number!r} score={getattr(ch,'score',0):.4f}")
        print(f"   {preview!r}")
    ranks = {
        gid: next((i for i, ch in enumerate(hits, 1) if ch.chunk_id == gid), None)
        for gid in gold
    }
    ranks_b = {
        gid: next((i for i, ch in enumerate(biased, 1) if ch.chunk_id == gid), None)
        for gid in gold
    }
    print("gold ranks hybrid:", ranks, "biased:", ranks_b)
