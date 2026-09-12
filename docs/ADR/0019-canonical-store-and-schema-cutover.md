# ADR-0019 — PostgreSQL + pgvector as the single canonical store; new schema in a new database

Status: accepted · Date: 2026-09-11 · Supersedes: 0001 (reaffirmed), 0005, 0007

## Context
The pre-rebuild schema (55 tables) mixed CAE investigation entities with a knowledge layer that had no regulation identity, version validity or lifecycle state. `Embedding.embedding` was an unsized `Vector()` with no ANN index; the same document existed as several `READY` revisions that all competed in retrieval.

## Decision
- PostgreSQL 16 + pgvector stays the only system of record: metadata, audit (ingestion runs/events, query traces) and vectors. No second vector database.
- The rebuilt schema lives in a **new database** (`safety_assistant`) with a **new Alembic chain** (`migrations/`, revision 0001). The pre-rebuild database is left untouched for recovery and is not migrated: chunking, sections and provenance are recomputed deterministically from the same source PDFs.
- `chunk_embeddings.embedding` is `vector(384)` with an HNSW cosine index. Changing the embedding model to another dimension is a schema migration by design; the provider factory refuses a model whose dimension does not match.

## Alternatives rejected
- Qdrant/other vector DB: no measured need — p50 retrieval is ~270 ms end-to-end on 21,910 chunks with HNSW; a second store adds consistency and operational cost.
- In-place data migration of the old tables: the chunker and normalizer are rewrites, so old chunks would be discarded anyway.

## Consequences
Retrieval applies lifecycle/temporal/authorization filters in SQL before ranking. Recovery of the old system is `git checkout pre-rebuild-baseline` plus the untouched `passive_safety` database.

## Evidence
`migrations/versions/0001_*.py` round-trips up/down/up; `tests/e2e`; `evals/results/retrieval_regulatory_v1_latest.json`.
