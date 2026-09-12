# ADR-0021 — Hybrid dense + BM25 retrieval fused by reciprocal rank, plus an exact-identifier leg

Status: accepted · Date: 2026-09-11 · Supersedes: 0009, 0012 (retrieval parts)

## Context
Regulatory questions mix paraphrase ("how far may the steering wheel move") with exact identifiers ("5.2.1.8", "HIC15"). Neither dense nor lexical retrieval alone covers both.

## Decision
Dense (pgvector HNSW, all-MiniLM-L6-v2) top-30 and BM25 (rank-bm25, light stemming, clause numbers kept as tokens) top-30, fused with RRF (k=60). When the query names a clause or annex, an exact-identifier leg (SQL on section paths / merged paths) joins the fusion with weight 2. Scope is applied in SQL before ranking; BM25 applies the same scope in memory over a per-generation cached index. Raw scores are never summed.

## Alternatives rejected
- Linear score combination: uncalibrated across BM25 and cosine.
- PostgreSQL FTS as the sparse leg: the baseline had already replaced `ts_rank` with real BM25 for identifier fidelity; kept.

## Consequences / evidence
On `regulatory_v1` (21,910 chunks): dense MRR 0.523, sparse 0.546, RRF 0.576, +reranker 0.597, full 0.636; R@10 0.793 → 0.901. The earlier repository claim that "RRF dilutes" did not hold on section-level truth.
