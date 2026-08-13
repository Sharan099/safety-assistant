# ADR-0015: Real cross-encoder reranker implemented and benchmarked — rejected as the default on latency, not quality

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md` §6 calls for a real cross-encoder
reranker, benchmarked against the existing `LexicalAuthorityReranker`
heuristic (`docs/ADR/0012`) on NDCG@10/MRR/Recall@10/latency, with an
explicit instruction: "Reject a candidate that does not improve ranking
quality on the domain golden set." §21's evaluation gates go further —
every candidate must clear **quality, latency, memory, and storage
together**, not quality alone.

`sentence-transformers`'s cross-encoder support needs `torch` (the same
disk risk `docs/ADR/0011` already measured). `fastembed` ships ONNX
cross-encoders with the same disk-safe profile as `docs/ADR/0014`'s
embedding provider — `Xenova/ms-marco-MiniLM-L-6-v2` (80 MB) and
`Xenova/ms-marco-MiniLM-L-12-v2` (120 MB).

## Decision

Implemented `CrossEncoderReranker` (`packages/retrieval/rerank.py`) and
benchmarked both variants (`evals/reranker_benchmark.py`, reranking real
RRF-fused candidates for the golden set) against the existing heuristic:

| Reranker | NDCG@10 | MRR | Recall@10 | Latency/query |
|---|---|---|---|---|
| lexical-authority-heuristic (current default) | 0.938 | 0.917 | 1.00 | 5.5 ms |
| **Xenova/ms-marco-MiniLM-L-6-v2** | **0.954** | **0.938** | 1.00 | **3,522.9 ms** |
| Xenova/ms-marco-MiniLM-L-12-v2 | 0.891 | 0.854 | 1.00 | 7,507.1 ms |

Two honest findings:

1. **L-6 genuinely improves ranking quality** — a real win, not noise.
2. **L-6 costs ~3.5 seconds per query on this CPU** (Intel i5-8250U). For an
   interactive investigation Copilot that already chains BM25 + dense +
   RRF + reranking + LLM drafting per turn, adding 3.5s to *every*
   knowledge-retrieving turn fails the doc's own "must remain practical on
   the development machine" / "no requirement for GPU inference" framing in
   practice, even though no GPU is technically required to run it.
3. **L-12 is rejected outright** — worse quality *and* twice the latency of
   L-6. Bigger is not automatically better; measured, not assumed.

`LexicalAuthorityReranker` **remains the production default**.
`CrossEncoderReranker` is real, tested, and available behind the same
`Reranker` Protocol (`get_cross_encoder_reranker()`, a cached singleton) —
not deleted, not hidden, just not the default — for a context that has
explicitly decided the latency is acceptable (offline evaluation batches,
or a future deployment on faster hardware where the 3.5s number would
change and could be re-measured).

## Consequences

- `retrieve()`'s reranking step stays fast (~5ms) — no change to the
  interactive Copilot's real-time responsiveness.
- The quality gap (NDCG@10 0.954 vs 0.938 — about 1.7%) is real but small
  at the current golden-set scale; revisit only if a larger/harder golden
  set shows a bigger, more consequential gap, per `TRD_LEVEL3.md` §45's
  "do not claim... before measuring" applied symmetrically (don't assume
  the gap stays this small at scale, either).
- If GPU inference or faster hardware becomes available, re-running
  `evals/reranker_benchmark.py` is the correct way to revisit this
  decision — not assuming it changed.
