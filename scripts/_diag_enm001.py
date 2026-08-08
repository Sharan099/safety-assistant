"""Diagnose enm_001 ranking under hybrid vs enumerative path."""

from __future__ import annotations

from retrieval.enumerative import (
    bias_chunks_for_enumerative_topic,
    classify_enumerative,
)
from retrieval.retrieve import hybrid_search, retrieve

q = "List every requirement related to doors in UN R95."
gold = {"c801fe9b9f12dbe8", "ce0be99f768f1c18", "f27d0b6751760843"}


def show(label: str, chunks: list) -> None:
    print(f"=== {label} ===")
    for i, c in enumerate(chunks[:15], 1):
        mark = "*" if c.chunk_id in gold else " "
        preview = (c.text or "").replace("\n", " ")[:70]
        print(f"{mark}{i:2d} {c.chunk_id} sec={c.section_number!r} {preview!r}")
    ranks = {
        cid: next((i for i, c in enumerate(chunks, 1) if c.chunk_id == cid), None)
        for cid in gold
    }
    print("gold ranks:", ranks)
    in5 = sum(1 for r in ranks.values() if r is not None and r <= 5)
    print(f"gold_in_top5={in5}/{len(gold)} recall@5={in5/len(gold):.2f}")


print("cls", classify_enumerative(q))
h = hybrid_search(q, regulation_id="UN-ECE-R95", top_k=40)
show("hybrid_search", h)
b = bias_chunks_for_enumerative_topic(h, question=q)
show("hybrid + topic bias", b)
r = retrieve(
    q,
    regulation_id="UN-ECE-R95",
    rewrite=False,
    do_rerank=True,
    small_to_big=True,
    top_k=20,
)
show("full retrieve enum path", r)
