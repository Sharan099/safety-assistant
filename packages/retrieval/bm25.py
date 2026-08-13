"""Real BM25 lexical retrieval — TRD_LEVEL3.md §16/§21, Instructions §17.

`packages/retrieval/search.py`'s FTS leg previously ranked candidates with
PostgreSQL's `ts_rank` over a `to_tsquery` — an honestly-labeled
*approximation*, not exact BM25 (`docs/ADR/0009`). `rank-bm25` is a ~50 KB
pure-Python package depending only on the already-installed `numpy` — no
disk/RAM risk the way `docling`/`sentence-transformers` are (`docs/ADR/0011`
doesn't apply here) — so there is no reason to keep the approximation once a
mathematically real implementation is this cheap. Instructions §17: "Do not
label a non-BM25 implementation as BM25" — this module makes the label true.

Builds an in-memory index from every `DocumentChunk` at call time. At the
corpus's current scale (regulations + solver manuals + real NHTSA reports —
low thousands of chunks) this is fast and always fresh. Revisit with a
persisted/incremental index only if a real corpus-growth benchmark shows the
rebuild cost mattering (TRD_LEVEL3.md §45: "do not claim... before
measuring" — no such measurement has been done, so no such claim is made
here either).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from packages.domain.knowledge import DocumentChunk

# Keeps '*' and '_' unlike relevance.py's tokenizer — CAE identifiers like
# "*MAT_024" and "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE" depend on both
# (TRD_LEVEL3.md §16's own worked examples), and BM25's exact-term matching
# is specifically what's supposed to carry those, not the dense/semantic leg.
_TOKEN_RE = re.compile(r"[a-z0-9_*]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Bm25Index:
    chunk_ids: list[uuid.UUID]
    bm25: BM25Okapi | None


def build_bm25_index(session: Session) -> Bm25Index:
    rows = session.query(DocumentChunk.id, DocumentChunk.content).all()
    if not rows:
        return Bm25Index(chunk_ids=[], bm25=None)
    chunk_ids = [r[0] for r in rows]
    corpus = [tokenize(r[1]) for r in rows]
    return Bm25Index(chunk_ids=chunk_ids, bm25=BM25Okapi(corpus))


def bm25_search(index: Bm25Index, query_text: str, *, limit: int) -> list[uuid.UUID]:
    if index.bm25 is None or not index.chunk_ids:
        return []
    tokens = tokenize(query_text)
    if not tokens:
        return []
    scores = index.bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    # A zero BM25 score means no term overlap at all — excluded rather than
    # handed a meaningless rank purely by virtue of existing in the corpus.
    return [index.chunk_ids[i] for i in ranked_indices if scores[i] > 0][:limit]
