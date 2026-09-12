# Retrieval design

Pipeline (`retrieval/service.py`):

1. **Scope parsing** (`domain/temporal/parse.py`): regulation keys (`UN R94`, `Regulation No. 94`, `FMVSS 208`), clause numbers (`5.2.1.8`, `paragraph …`), annexes, as-of dates (`as of 2019-06-01`, `in 2015`, `before 2021`), intent hints. Deterministic regexes with unit tests.
2. **SQL scope** (`retrieval/base.py`): version status (ACTIVE, or ACTIVE|SUPERSEDED for historical), validity window on the effective date, regulation keys, kinds, authority levels, principal data classes. Applied before any scoring.
3. **Dense leg**: pgvector cosine over `chunk_embeddings` (HNSW, m=16, ef_construction=64), top-30.
4. **Sparse leg**: rank-bm25 over all retrievable chunks; tokenizer keeps decimal clause numbers and `*MAT_024`-style identifiers, applies light suffix stemming (`doors→door`, `categories→category`); index cached per corpus generation and scoped in memory; top-30 after scoping.
5. **Exact leg** (when a clause/annex is named): section path or merged-path match, weight 2 in fusion.
6. **RRF** (k=60) over rank lists; never raw score sums.
7. **Rerank**: heuristic (fused score + literal overlap + authority tier + normative bonus for requirement-seeking queries) by default; cross-encoder opt-in; failure degrades to fused order.
8. **Guard + diversify**: authority allowlist, literal relevance floor (1 shared term for ≤2-term queries, else 2), content dedup, per-version cap that is scope-aware (no cap when only one version is in play).
9. **Evidence bundle** (`retrieval/context.py`): stable ids E1…En, parent section text, up to two resolved cross-references, token budget (6,000 estimated tokens), full provenance per item.

## Candidate sizes and tuning

Defaults: dense 30, sparse 30, final k 10, budget 6,000 tokens. Changes must come with a before/after row in `evals/results/`.

## Measured iteration record (`regulatory_v1`, full leg)

| Change | MRR | R@10 |
|---|---|---|
| first run (18.6k chunks) | 0.627 | 0.761 |
| + light stemming, short-query guard, normative-aware rerank | 0.657 | 0.825 |
| + scope-aware diversification cap (was dropping correct clauses on single-regulation scope) | 0.664 | 0.901 |
| final corpus (21.9k chunks, chunker 2.0.2) | 0.636 | 0.901 |

Weakest slices: numeric thresholds where an annex computation procedure competes with the body clause (MRR ~0.5), multi-clause reasoning (n=1). Candidate experiments (not in production): cross-encoder default (E3), contextual chunk headers with section titles (E4), fusion weights (E6).
