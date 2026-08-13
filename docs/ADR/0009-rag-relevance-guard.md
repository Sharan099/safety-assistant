# ADR-0009: RAG relevance guard — fixed FTS's AND-only matching, added a lexical relevance floor

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`PRD_COPILOT_UPDATE.md` §8 and `CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md` Phase 8
report a retrieval-quality problem observed in real use: documentary
retrieval surfaced clearly unrelated material (an AES/Rijndael passage) as
"evidence." Reproducing this against the real corpus surfaced two distinct,
compounding root causes — not one.

## Finding 1 — `plainto_tsquery` ANDs every query term together

Querying `retrieve(session, "restraint configuration belt force limiter
webbing revision")` — a realistic investigation-style question — returned
**zero** full-text hits, despite UN_R94 discussing belts and force limiters
extensively. `plainto_tsquery` (and `websearch_to_tsquery`) require every
lexeme to appear in the same chunk; a 6-7 term query essentially never
survives that against ~400-word chunks. Every result the old code returned
came entirely from the vector leg.

**Fix:** `full_text_search` now OR-joins tokenized query terms into an
explicit `to_tsquery('term1 | term2 | ...')`. `ts_rank` still does the work
of ranking better matches higher; OR just means a partial match is a
candidate at all instead of being silently excluded.

## Finding 2 — the interim embedding provider can score irrelevant content nonzero

With FTS effectively contributing nothing to many real queries, results
were vector-search-only — and `HashingEmbeddingProvider` (docs/ADR/0007) has
no semantic understanding, only hashed word-overlap. It can and does surface
a chunk that shares no real topic with the query but collides enough hash
buckets to get a nonzero cosine score.

**Fix:** `packages/retrieval/relevance.py` adds a lexical relevance floor
*independent of the embedding*: a chunk must share at least 2 significant
(non-stopword, length > 2) terms with the query, literally — not via a
hashed vector. Combined with an authority-level allowlist and dedup (exact
content + max 3 chunks/document), `retrieve()` now never returns a raw
top-k list; everything passes the guard first.

## Consequences

- `evals/retrieval_eval.py`'s golden set should be re-run after this change
  (Recall@5/MRR may shift now that FTS actually contributes candidates).
- The lexical floor is deliberately crude (it's the same category of fix as
  ADR-0007's embedding — good enough to stop egregious false positives, not
  a substitute for a real relevance model). Revisit together with the
  embedding-model benchmark in TRD.md §20.
- `min_shared_terms` is a `retrieve()` parameter, not a hardcoded constant,
  so callers needing looser/stricter matching (e.g. the Copilot vs. the
  Knowledge search page) can tune it without touching this module.
