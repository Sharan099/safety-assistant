# ADR-0012: Real BM25, a lightweight reranker, and structured search's scope

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`docs/ADR/0009` already flagged that `packages/retrieval/search.py`'s FTS
leg used PostgreSQL's `ts_rank` — an honestly-labeled *approximation*, not
exact BM25. `TRD_LEVEL3.md` §16/§21 makes replacing this with a real
implementation, or explicitly documenting the approximation, a Level-3
requirement: "Do not label a non-BM25 implementation as BM25."

`TRD_LEVEL3.md` §15/§20 also make hybrid retrieval — BM25 + Dense +
Structured → RRF → Reranker → Gate — a **mandatory** Level-3 requirement.
But `PRD_LEVEL3.md` §14 states plainly: "Structured search is complementary
to RAG," and the existing (pre-Level-3) tool list in `PRD_LEVEL3.md` §26
already keeps `retrieve_knowledge` and `retrieve_structured_cae` as two
separate tools — the same document contradicts its own architecture
diagram if the diagram is read as "literally fuse structured rows and text
chunks into one RRF-ranked list."

## Decision

**BM25**: `rank-bm25` (`BM25Okapi`) replaces `ts_rank`. It is a ~50 KB
pure-Python package depending only on the already-installed `numpy` — none
of `docs/ADR/0011`'s disk-risk reasoning applies, so there is no reason to
keep an approximation once a correct implementation is this cheap.
`packages/retrieval/bm25.py` builds an in-memory index from every
`DocumentChunk` per call (documented as revisit-if-corpus-growth-shows-it-
matters, not measured yet).

**Reranker**: `packages/retrieval/rerank.py`'s `LexicalAuthorityReranker`
runs after RRF fusion, before the relevance/authority gate — the real
architecture slot, with a dependency-free scoring function (RRF score +
literal query-term overlap bonus + authority-tier bonus) instead of a real
cross-encoder, for the same disk-headroom reason as `docs/ADR/0011`. Same
`Protocol`-swap posture as `docs/ADR/0007`.

**Structured search scope**: Given the internal contradiction above, the
least-disruptive reading consistent with Level-3 requirements is PRD_LEVEL3
§14's own explicit prose, not a literal interpretation of the architecture
diagram: `packages/retrieval/structured.py` (deck/part/material/contact/
control/database/include queries, `docs/ADR` — see the CAE schema commit)
remains a **separate, complementary** retrieval capability the agent can
call as its own tool, not a candidate source fused into `retrieve()`'s RRF
ranking alongside text chunks. Fusing a `CaePart` row and a text chunk into
one rank without a shared, principled scoring space would be fabricated
precision, not real hybrid retrieval.

## Consequences

- `retrieve()`'s pipeline is now genuinely BM25 (not an approximation) +
  Dense → RRF → Reranker → Guard, satisfying TRD_LEVEL3.md §20's mandatory
  three-stage shape for the *text-knowledge* retrieval path.
- Structured CAE questions ("which material is used by Part 1042?") are
  answered by calling `packages/retrieval/structured.py` directly (a new
  `retrieve_structured_cae`-style agent tool, matching the pre-existing tool
  list) — not by expecting `retrieve_knowledge` to surface them.
- `RetrievedChunk` gained a `rerank_score` field (distinct from
  `fused_score`) so both scores stay inspectable — TRD_LEVEL3.md §28
  retrieval observability.
- If a future evaluation genuinely needs BM25/Dense/Structured fused into
  one ranked, citable list, that requires designing a real shared relevance
  space first (e.g. converting structured facts into synthetic "chunks"
  with their own text representation) — not attempted here without that
  design and a measurement backing it, per `TRD_LEVEL3.md` §45's own rule.
