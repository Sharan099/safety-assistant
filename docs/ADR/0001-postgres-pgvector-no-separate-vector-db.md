# ADR-0001: PostgreSQL + pgvector, no separate vector database

- **Status:** Accepted
- **Date:** 2026-08-12

## Context

The system needs relational storage for the domain/investigation model
(runs, evidence, hypotheses, ...) plus vector search over document chunks and
historical-case embeddings. Development hardware is an 8 GB RAM / 2 GB GPU
Windows laptop (`ENVIRONMENT_SETUP.md` §1).

## Decision

Use PostgreSQL with the `pgvector` extension for both relational and vector
data. Do not introduce Qdrant, Weaviate, Milvus, or Elasticsearch in V1.

## Rationale

- One service to run/operate on constrained hardware instead of two.
- JSONB covers flexible engineering metadata; FTS covers keyword retrieval.
- Transactions span domain writes and evidence/provenance writes.
- `TRD.md` §7 and `BOOTSTRAP_PROMPT.md` §13 mandate this explicitly.

## Consequences

- Retrieval quality (recall/MRR) must be benchmarked (Phase 12) before
  considering a dedicated vector DB. Only a measured bottleneck justifies
  revisiting this decision.
- Large numerical time-series data (signals) is explicitly kept out of
  Postgres — Parquet + DuckDB instead (`TRD.md` §8).
