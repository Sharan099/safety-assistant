# ADR-0007: Deterministic hashing embedding as the interim EmbeddingProvider

- **Status:** Accepted (interim — expected to be superseded)
- **Date:** 2026-08-13

## Context

`TRD.md` §20 requires an `EmbeddingProvider` interface and explicitly says
not to commit to a model before benchmarking (CPU speed, RAM, quality,
dimensions, licensing, multilingual support). But `packages/retrieval`
(FTS + pgvector + RRF, `TRD.md` §19) needs *some* embedding to exist end to
end — vector search, RRF fusion, and the retrieval API are otherwise
untestable.

Installing a real sentence-embedding model today means `sentence-transformers`
+ `torch`, several hundred MB to a few GB, on top of everything else already
installed on the 8 GB RAM dev machine (`ENVIRONMENT_SETUP.md` §1: "no local
large language model," "keep background services minimal").

## Decision

`packages/retrieval/embeddings.py` ships `HashingEmbeddingProvider`: a
deterministic, dependency-free, fully offline feature-hashed bag-of-words
embedding (SHA-256 token hashing into a 256-dim signed, L2-normalized
vector). It is the default `EmbeddingProvider` until a real model is
benchmarked and chosen.

This is explicitly a **placeholder for mechanism, not quality**: it makes
pgvector storage, cosine search, and RRF fusion against FTS real and
testable now. Retrieval quality is word-overlap only — no semantics, no
synonym handling.

## Consequences

- Vector search results will be noticeably worse than a real embedding
  model until this is replaced. That's expected and documented, not a bug.
- The `EmbeddingProvider` `Protocol` is the only contract `packages/retrieval`
  depends on; swapping in a real model (once TRD.md §20's benchmark picks
  one) requires no changes above `packages/retrieval/embeddings.py`.
- `Embedding.model_name`/`model_version` are part of the row, so old and new
  provider embeddings can coexist during a migration — `index_chunks()`
  already keys "already embedded" off `(chunk, model_name, model_version)`.
- The `embeddings.embedding` column has no fixed vector dimension yet
  (`docs/ADR/` — see `packages/domain/knowledge.py`); a real model's
  dimension should get a dedicated migration with an ivfflat/hnsw index once
  chosen, per `TRD.md` §20.
