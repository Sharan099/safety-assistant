import json
from pathlib import Path


def avg(key, rows):
    return sum((r.get(key) or 0.0) for r in rows) / max(1, len(rows))


def has_chunk_gold(c):
    for g in c.get("gold") or []:
        s = str(g)
        if "::" in s or s.startswith("Annex"):
            continue
        if s.replace(".", "").replace("/", "").isdigit():
            continue
        if len(s) >= 12:
            return True
    return False


def subset(d, label):
    cases = d["retrieval"]["cases"]
    with_gold = [c for c in cases if c.get("gold")]
    chunk_gold = [c for c in cases if has_chunk_gold(c)]
    print(label, "all n", len(cases), "mrr", round(avg("mrr", cases), 4), "r@5", round(avg("recall@5", cases), 4))
    print(
        label,
        "chunk_gold n",
        len(chunk_gold),
        "mrr",
        round(avg("mrr", chunk_gold), 4),
        "r@5",
        round(avg("recall@5", chunk_gold), 4),
        "p@5",
        round(avg("precision@5", chunk_gold), 4),
        "ndcg@10",
        round(avg("ndcg@10", chunk_gold), 4),
    )
    sample = [
        "cmp_004",
        "cmp_001",
        "cmp_003",
        "num_003",
        "num_005",
        "num_004",
        "xrg_001",
        "enm_001",
        "fac_003",
        "fig_001",
    ]
    by = {c["id"]: c for c in cases}
    for cid in sample:
        o = by.get(cid) or {}
        print(
            f"  {cid}: mrr={o.get('mrr')} r@5={o.get('recall@5')} "
            f"hits={len(o.get('hit_chunk_ids') or [])}"
        )


old = json.loads(
    Path("eval/results/stage5_rebuild-retrieval-only-20260807T232116Z.json").read_text(
        encoding="utf-8"
    )
)
new = json.loads(
    Path("eval/results/remap-v2-content-grounded-retrieval-only-20260808T065225Z.json").read_text(
        encoding="utf-8"
    )
)
print("OLD scorecard", old.get("scorecard"))
print("NEW scorecard", new.get("scorecard"))
subset(old, "OLD")
print("----")
subset(new, "NEW")
