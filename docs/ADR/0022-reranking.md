# ADR-0022 — Heuristic reranker by default, cross-encoder opt-in

Status: accepted · Date: 2026-09-11 · Reaffirms: 0015

## Context
A cross-encoder (Xenova/ms-marco-MiniLM-L-6-v2) improves ordering but was measured at ~3.5 s/query on the target CPU.

## Decision
The default reranker is the lexical + authority heuristic extended with a normative-clause bonus for requirement-seeking queries. `RERANKER=cross_encoder` selects the cross-encoder through the same `Reranker` protocol. Reranker failure degrades to fused order and is marked in `versions.degraded`.

## Evidence
`hybrid_rrf` → `hybrid_rrf_rerank`: MRR 0.576 → 0.597, nDCG@10 0.614 → 0.639, +6 ms p50. Pre-rebuild cross-encoder benchmark: `evals/baselines/pre-rebuild_reranker_benchmark.json`.
