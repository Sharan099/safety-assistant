# Architecture

One Python package (`src/safety_assistant/`), one PostgreSQL database, one Next.js frontend.

```
src/safety_assistant/
  config/        Settings profiles (development | test | production); production refuses fakes
  persistence/   SQLAlchemy models (14 tables), engine/session; migrations/ (Alembic chain 0001…)
  domain/        regulations/lifecycle (state machine), temporal/parse (scope, clause ids, as-of dates)
  ingestion/     sources (registry allowlist) · fetch (blob stores, SSRF-safe http) · validation
                 parse (DocumentParser contract, PyMuPDF impl) · normalize (clause tree, cover page)
                 chunk (structural) · index (embed with content-hash reuse) · diff (version diff)
                 workflows/ingest.py (lifecycle orchestration, idempotency, quarantine, activation)
  retrieval/     filters (scope, relevance floor) · base (scoped SQL) · dense · sparse (BM25)
                 fusion (RRF) · rerank · context (evidence bundle, parent/xref expansion) · service
  providers/     embeddings · rerankers · llm — Protocols + real implementations + test fakes
  generation/    schemas (contract) · grounding (gate) · citations (validation) · prompts · service
  agents/        state/budget · tools (typed) · graph (bounded LangGraph)
  api/           routes (health, query, versions, admin) · dependencies (auth, oidc) · middleware
  observability/ JSON logging · OpenTelemetry · Prometheus metrics
  security/      injection signals
  evaluation/    dataset schema · IR metrics · per-leg runner
  cli.py         ingest | migrate | eval-retrieval | ready
```

## Data flow

1. **Registry** (`knowledge/00_registry/sources.yaml`) declares every source: identity, kind, jurisdiction, authority level, data class, official URI, hash, and the version's own label/series/dates.
2. **Ingestion** (`workflows/ingest.py`) walks the lifecycle; every stage writes an `ingestion_events` row. Bytes are stored content-addressed in the blob store; structure goes to `sections/tables/figures/cross_references`; `chunks` are deterministic (uuid5 of version, ordinal, content hash); `chunk_embeddings` reuse vectors for identical content across versions of the same regulation. `VERIFIED → ACTIVE` supersedes the previous version atomically.
3. **Query** (`generation/service.py` → `agents/graph.py`): scope is parsed deterministically, authorization narrows data classes, `retrieval/service.py` produces an evidence bundle, the gate decides (abstain / rewrite once / proceed), the LLM (if configured and cleared) answers under schema, citations and numbers are validated, a `query_traces` row is written.
4. **API/UI**: `/api/v1/ask` returns mode, answer, typed claims with evidence ids, citations with version/validity/source hash, warnings, and the full evidence; `/api/v1/evidence/{chunk_id}` resolves any citation to its exact provenance.

## Boundaries and invariants

- Only `ACTIVE` versions are retrievable for current questions; `ACTIVE|SUPERSEDED` for as-of questions (`domain/regulations/lifecycle.py`).
- IDs, hashes, dates, authorization, budgets and citation resolution are code; the LLM only synthesises.
- No fake providers outside `APP_ENV=test`; readiness fails if embeddings cannot load; an LLM outage degrades to evidence-only.
- CPU-bound work (parsing, embeddings, reranking) runs in worker threads; the event loop stays responsive (tested).

## Traceability chain

`answer.claims[].evidence_ids` → `evidence[].chunk_id` → `chunks` → `sections` (path, page) → `regulation_versions` (label, status, valid_from/to, parser/chunker/index versions, amendments) → `source_artifacts` (sha256, storage_uri, source_uri, retrieved_at) → `regulations` (key, jurisdiction, authority). `GET /api/v1/evidence/{chunk_id}` returns the whole chain.
