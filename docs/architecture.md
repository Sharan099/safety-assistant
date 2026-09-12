# Architecture

One Python package (`src/safety_assistant/`), one PostgreSQL database, one queue worker process, one Next.js frontend.

```
src/safety_assistant/
  config/        Settings profiles (development | test | production); production refuses fakes, dev login, weak session secrets
  persistence/   SQLAlchemy models (25 tables), engine/session; migrations/ (Alembic chain 0001…0004)
  domain/        regulations/lifecycle (state machine) · temporal/parse · documents (scope enum, display status, public errors)
  identity/      users, organizations, memberships, workspaces, preferences, browser sessions (HS256 cookie), audit events
  documents/     upload (bounded, deduped, scoped), listings through the authorization predicate, jobs, archive, promotion
  workers/       ingestion queue worker (FOR UPDATE SKIP LOCKED claim, bounded retries with backoff, public error codes)
  conversations/ persistent threads: messages, citations, source scope, prior turns as wording context only
  ingestion/     sources (registry allowlist) · fetch (blob stores, SSRF-safe http) · validation
                 parse (DocumentParser contract, PyMuPDF impl) · normalize (clause tree, cover page)
                 chunk (structural) · index (embed with content-hash reuse) · diff (version diff)
                 workflows/ingest.py (ingest_source for registry batches, ingest_version for the queue; shared stage pipeline)
  retrieval/     authz (document-level predicate: SQL + in-memory mirror) · filters · base (scoped SQL) · dense · sparse (BM25)
                 fusion (weighted RRF) · rerank (heuristic or cross-encoder, top-N cap) · context (evidence bundle) · service
  providers/     embeddings · rerankers · llm — Protocols + real implementations + test fakes
  generation/    schemas (contract) · grounding (gate) · citations (validation) · prompts (grounded_v2) · service
  agents/        state/budget · tools (typed) · graph (bounded LangGraph; conversation context is data, never evidence)
  api/           routes (health, query, versions, admin, me, conversations, documents) · dependencies (auth, oidc) · middleware
  observability/ JSON logging · OpenTelemetry · Prometheus metrics
  security/      injection signals
  evaluation/    dataset schema · IR metrics · per-leg retrieval runner · end-to-end generation runner
  cli.py         ingest | migrate | eval-retrieval | ready | worker | users add
frontend/        Next.js App Router: /login, /app/{home,chat,chat/[id],documents,documents/upload,documents/[id],ingestion,settings,admin}
```

## Tenancy and authorization

`organizations → memberships (engineer | knowledge_admin | auditor | org_admin) → users`, `workspaces → workspace_memberships`.
A `regulations` row is the logical *document* and carries `scope ∈ {AUTHORITATIVE_ORG, WORKSPACE, PRIVATE_USER}`,
`organization_id`, `workspace_id`, `owner_user_id`, `archived_at`. `retrieval/authz.py` defines the single
predicate — organization membership for authoritative sources, workspace membership for workspace documents,
ownership for private documents, intersected with the user's *selected* source scopes — and it is evaluated in
SQL before any ranking (dense/exact legs, listings, evidence lookup) and mirrored in the BM25 pre-filter.
A unit test proves the two evaluators agree. Principals without a user identity (API keys, evaluation scripts)
see authoritative documents only.

Browser sessions are HttpOnly `session` cookies (HS256, `typ=session`); mutating cookie-authenticated requests
must carry `X-Requested-With`. OIDC bearer tokens resolve to a persisted user when one exists
(`safety-assistant users add`). `dev-login` exists only when `DEV_LOGIN_ENABLED=true`, never in production.

## Data flow

1. **Registry sources** (`knowledge/00_registry/sources.yaml`) declare identity, kind, authority level, data class, hash and
   version dates. `safety-assistant ingest` runs them synchronously; `POST /api/v1/admin/ingest` stages the bytes and enqueues.
2. **Uploads** (`POST /api/v1/documents`, multipart, ≤ `INGEST_MAX_FILE_BYTES`, PDF magic, safe filename) create a document
   (`PRIVATE_USER` by default or a workspace the caller belongs to), a `DISCOVERED` version over a content-addressed artifact,
   and an `ingestion_jobs` row. Identical bytes re-uploaded by the same owner return the existing ids.
3. **Worker** (`safety-assistant worker`) claims due jobs on the database clock, runs the same stage pipeline as the registry
   path (validate → parse → normalize → chunk → embed → index → verify → activate), records every stage in `ingestion_events`,
   and finishes the job as SUCCEEDED / QUEUED-with-backoff / FAILED / QUARANTINED with a public error code and a diagnostic
   reference (`ingestion_runs.id`). Only `ACTIVE` (= READY) versions are retrievable.
4. **Query** (`/api/v1/ask` or `/api/v1/conversations/{id}/messages`): scope parsed deterministically; authorization predicate +
   data classes narrow the universe; `retrieval/service.py` fuses dense + BM25 + exact legs (RRF, dense weight 0.75), reranks,
   guards and expands; the gate decides (abstain / rewrite once / proceed); the LLM answers under schema with prior turns
   rendered as `<conversation_context>` data; citations and numbers are validated; a `query_traces` row is written; the
   conversation stores both turns and the citations (label, section, version, hash) denormalised.
5. **UI**: three-pane workbench — investigations, messages with citation markers, evidence panel showing clause/version/page/
   validity/scope/excerpt for every cited (and retrieved-but-uncited) chunk; document library with real stage timeline.

## Boundaries and invariants

- Authorization before ranking; READY-only retrieval; failed/quarantined/archived versions are never retrievable.
- Uploads never become authoritative automatically — `POST /documents/{id}/promote` needs `document:promote` and is audited.
- Chat history and preferences are never evidence: citations can only reference evidence retrieved in the current request.
- No fake providers outside `APP_ENV=test`; readiness fails if embeddings cannot load; an LLM outage degrades to evidence-only.
- Expensive work is asynchronous (queue worker) or offloaded to threads; the event loop stays responsive (tested).

## Traceability chain

`answer.claims[].evidence_ids` → `evidence[].chunk_id` → `chunks` → `sections` (path, page) → `regulation_versions`
(label, status, validity, parser/chunker/index versions) → `source_artifacts` (sha256, storage_uri, source_uri) →
`regulations` (key, scope, organization/workspace/owner). `GET /api/v1/evidence/{chunk_id}` returns the whole chain,
subject to the same authorization predicate.
