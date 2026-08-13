# ADR-0014: Real semantic embeddings via fastembed (ONNX), replacing the hashing placeholder

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`docs/ADR/0007` shipped `HashingEmbeddingProvider` as an explicitly interim,
dependency-free placeholder, and said measuring, not assuming, should
decide when to replace it. `evals/level3_hybrid_eval.py` (ADR-0012) then
measured exactly that: on the real golden set, the hashing placeholder's
dense leg *dragged down* fused RRF quality relative to BM25 alone (MRR
0.677 vs 0.917). `PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md` §5 calls replacing it
"P0" — the highest-priority remaining gap.

The doc's suggested candidates (BGE-M3, an E5-family model, a Qwen-
embedding model) are all >2 GB via the `sentence-transformers`/`torch`
stack — the exact disk-risk `docs/ADR/0011` already measured (12 GB free
disk) and deferred Docling/a real reranker for. `fastembed` uses ONNX
Runtime instead: no `torch`, no `transformers`, and its small quantized
models (67-90 MB) carry none of that risk — `uv add fastembed` pulled 14
packages, no `torch`, disk headroom unaffected at the GB-resolution `df -h`
measurement.

## Decision

Benchmarked two small, real, hardware-appropriate candidates against the
hashing baseline (`evals/embedding_benchmark.py`, results in
`evals/results/embedding_benchmark.json`) — substituting the doc's named
>2 GB candidates with small ones per the *same* doc's own hardware rules
(§20: "no giant local model merely to claim SOTA"):

| Provider | dim | Recall@5 | Recall@10 | MRR | Index time (475 real chunks) |
|---|---|---|---|---|---|
| hashing-bow (baseline) | 256 | 0.75 | 0.88 | 0.249 | 0.7s (already embedded) |
| BAAI/bge-small-en-v1.5 | 384 | 1.00 | 1.00 | 0.729 | 143.8s |
| **sentence-transformers/all-MiniLM-L6-v2** | 384 | 1.00 | 1.00 | **0.838** | 22.9s |

`all-MiniLM-L6-v2` wins on the deciding metric (MRR) and indexes 6x faster
than the other real candidate. `FastEmbedProvider`
(`packages/retrieval/embeddings.py`) becomes the production default for
both `vector_search()`/`retrieve()` and `index_chunks()`, replacing
`HashingEmbeddingProvider` in that role. `HashingEmbeddingProvider` is kept
as the explicit "mock" tier (deterministic, no network/model download —
used by tests that don't need real semantic behavior).

A `get_default_embedding_provider()` process-wide singleton
(`functools.lru_cache`) avoids reloading the ~1-2s ONNX model on every
`retrieve()` call — real, measured cost, not a style preference.

**Remote tier**: not implemented. `LLM_BASE_URL`/`LLM_API_KEY` (`docs/ADR/0003`)
point at FreeLLMAPI, but no `.env` exists in this environment and no
credentials are configured — building an untested remote `EmbeddingProvider`
implementation would risk exactly the kind of silent, unverified bug the
"never silently substitute a fake capability" rule exists to prevent. The
`EmbeddingProvider` `Protocol` is the only integration point a real remote
implementation needs; nothing above this module changes when one is added.

## Consequences

- Every existing chunk was re-embedded under `sentence-transformers/all-MiniLM-L6-v2`
  (via `evals/embedding_benchmark.py`'s own `index_chunks()` call, which is
  the same idempotent production code path) — no separate migration script
  needed.
- `docs/ADR/0012`'s honest finding ("BM25 alone beats hybrid, given the
  interim embedding provider") is now stale as a *current* production claim
  — the RRF fusion should be re-measured with the new provider
  (`evals/level3_hybrid_eval.py`) and the result reported honestly whichever
  way it comes out, not assumed.
- The `embeddings.embedding` pgvector column still has no fixed dimension
  (`docs/ADR/0007`'s deferred point) — an ivfflat/hnsw ANN index needs one.
  At today's real corpus scale (~500 chunks) brute-force cosine search is
  fast enough that this remains correctly deferred, not silently ignored.
- `HashingEmbeddingProvider`'s old default role is gone; any code
  constructing it explicitly for production behavior (there was none found)
  would need updating — verified by grep across `packages/`/`apps/`/`scripts/`.
