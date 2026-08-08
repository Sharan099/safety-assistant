"""Audit HPC acronym expansion + dense/sparse/hybrid ranks for R95 limit Q."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from qdrant_client import models as qm

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from ingestion.embed_upsert import Embedder
from retrieval.acronyms import expand_acronyms
from retrieval.retrieve import (
    BM25_MODEL,
    DEFAULT_COLLECTION,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    _qdrant_search_params,
    _regulation_filter,
    get_qdrant_client,
    hybrid_search,
    retrieve,
)
from retrieval.value_limit import criteria_focused_subquery, is_value_vs_limit_query


QUESTION = "What is the HPC limit in UN R95?"


def _show(label: str, points) -> None:
    print(f"\n=== {label} ===")
    for i, h in enumerate(points, 1):
        p = h.payload or {}
        txt = " ".join((p.get("text") or "").split())[:140]
        print(
            f"{i}. score={float(h.score or 0):.4f} "
            f"reg={p.get('regulation_id')} sec={p.get('section_number')} "
            f"id={p.get('chunk_id')} | {txt}"
        )


def main() -> None:
    expanded = expand_acronyms(QUESTION)
    print("QUESTION:", QUESTION)
    print("EXPANDED:", expanded)
    print("value_vs_limit?", is_value_vs_limit_query(QUESTION))
    print("criteria subquery:", criteria_focused_subquery(QUESTION))

    client = get_qdrant_client()
    embedder = Embedder()
    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    dense = embedder.embed([expanded])[0]
    params = _qdrant_search_params()

    for rid, label in [(None, "no filter"), ("UN-ECE-R95", "R95 filter")]:
        qfilter = _regulation_filter(rid)
        r_dense = client.query_points(
            collection_name=collection,
            query=dense,
            using=DENSE_VECTOR,
            query_filter=qfilter,
            limit=8,
            with_payload=True,
            search_params=params,
        )
        _show(f"DENSE ({label})", r_dense.points)

        r_sp = client.query_points(
            collection_name=collection,
            query=qm.Document(text=expanded, model=BM25_MODEL),
            using=SPARSE_VECTOR,
            query_filter=qfilter,
            limit=8,
            with_payload=True,
        )
        _show(f"SPARSE/BM25 ({label})", r_sp.points)

        # Qdrant RRF hybrid (same as hybrid_search)
        r_hy = client.query_points(
            collection_name=collection,
            prefetch=[
                qm.Prefetch(
                    query=dense,
                    using=DENSE_VECTOR,
                    limit=40,
                    filter=qfilter,
                    params=params,
                ),
                qm.Prefetch(
                    query=qm.Document(text=expanded, model=BM25_MODEL),
                    using=SPARSE_VECTOR,
                    limit=40,
                    filter=qfilter,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=8,
            with_payload=True,
            search_params=params,
        )
        _show(f"QDRANT RRF HYBRID ({label})", r_hy.points)

    print("\n=== hybrid_search() helper ===")
    for i, c in enumerate(hybrid_search(QUESTION, top_k=8), 1):
        txt = " ".join((c.text or "").split())[:140]
        print(
            f"{i}. score={c.score:.4f} reg={c.regulation_id} "
            f"sec={c.section_number} id={c.chunk_id} | {txt}"
        )

    print("\n=== retrieve() full pipeline (rerank on, no small-to-big) ===")
    chunks = retrieve(
        QUESTION,
        top_k=8,
        rewrite=True,
        do_rerank=True,
        small_to_big=False,
    )
    for i, c in enumerate(chunks, 1):
        txt = " ".join((c.text or "").split())[:140]
        print(
            f"{i}. score={c.score:.4f} reg={c.regulation_id} "
            f"sec={c.section_number} id={c.chunk_id} | {txt}"
        )


if __name__ == "__main__":
    main()
