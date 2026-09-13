# ADR-0029 — v2 product domain: identity, scoped documents, async jobs, conversations; authorization predicate

Status: accepted (D-003, D-012 confirmed by owner 2026-09-12; Phase C implemented) · Date: 2026-09-12 · Extends: 0019 (canonical regulatory schema), 0027 (security policy)

## Context

The v1 rebuild delivers a verified regulatory corpus (`regulations` → `regulation_versions` → `sections`/`chunks`), hybrid retrieval with SQL scope before ranking, grounded generation and role-based auth. It has **no persisted users, no document ownership, no user uploads, no async jobs, no conversations** (`docs/rebuild/STATE.md`). `docs/product/02_TRD.md` requires all of them. This ADR is the schema/authorization/API plan for Phases C–D; it changes no code.

## Decision 1 — Reuse the corpus tables as the document model (no rename)

| TRD entity | Implementation |
|---|---|
| `Document` | existing `regulations` row (logical identity). New columns: `scope`, `organization_id`, `workspace_id`, `owner_user_id`. Domain code and API call it *document*; the table name stays (a rename buys nothing and costs a data migration + every query). |
| `DocumentVersion` | existing `regulation_versions` (lifecycle, lineage, `activated_at`, `source_artifact_id` already present). |
| `Chunk` | existing `chunks` + `chunk_embeddings`. |
| `IngestionJob` | **new** `ingestion_jobs` (queue row); existing `ingestion_runs`/`ingestion_events` remain the execution log (job 1 → runs n, one per attempt). |

Uploaded documents get `regulation_key = "DOC-<12 hex of sha256>"`, `kind` from the upload form (`REGULATION|STANDARD|TECHNICAL_REPORT|MANUAL|PROJECT_DOCUMENT`), `authority_level = "REFERENCE"` (never `AUTHORITATIVE`), `data_class = "CONFIDENTIAL"` for `PRIVATE_USER`/`WORKSPACE` scope (drives the existing provider data-class policy: confidential evidence is not sent to a provider that is not allowed to see it — degrade to `EVIDENCE_ONLY`).

## Decision 2 — Lifecycle vocabulary: internal `VersionStatus` kept, spec vocabulary is the API view

The proven state machine (`domain/regulations/lifecycle.py`: DISCOVERED → DOWNLOADED → VALIDATED → PARSED → NORMALIZED → CHUNKED → INDEXED → VERIFIED → ACTIVE | SUPERSEDED | QUARANTINED | FAILED) is a refinement of the spec's list. A pure mapping in `domain/documents.py` renders it:

```
DISCOVERED, DOWNLOADED → UPLOADED      NORMALIZED, CHUNKED → CHUNKING
VALIDATED (job running) → VALIDATING   INDEXED → INDEXING (embedding happens inside)
PARSED → PARSING                        VERIFIED → VERIFYING
ACTIVE → READY                          SUPERSEDED → ARCHIVED
QUARANTINED, FAILED → themselves
```

The *job* carries `stage` (the spec's stage names) and progress; the *version* carries truth. `READY` in every spec sentence means `VersionStatus.ACTIVE`; `RETRIEVABLE_CURRENT = {ACTIVE}` is unchanged, so "only READY versions are retrievable" is already enforced by `scoped_statement` and `VersionMeta.in_scope`.

## Decision 3 — Schema additions (forward migrations, all additive except one backfilled NOT NULL)

### `0002_identity`
```
organizations(id, name, created_at)
users(id, email UNIQUE, display_name, status, created_at, last_login_at)
memberships(user_id, organization_id, role, PK(user_id, organization_id))          role ∈ engineer|knowledge_admin|auditor|org_admin
workspaces(id, organization_id, name, created_by, created_at, archived_at)         UNIQUE(organization_id, name)
workspace_memberships(workspace_id, user_id, role, PK(workspace_id, user_id))       role ∈ member|owner
user_preferences(user_id PK, default_workspace_id, answer_density, preferred_language, ui_theme, updated_at)
audit_events(id, actor_user_id, organization_id, action, resource_type, resource_id, request_id, metadata JSONB, created_at)   index (organization_id, created_at)
```
Data: insert organization `default` (fixed UUID in the migration so it is idempotent). Downgrade drops all seven tables.

### `0004_document_scope_and_jobs` (Phase D; numbered after conversations, which shipped first)
```
regulations + scope TEXT NOT NULL DEFAULT 'AUTHORITATIVE_ORG'
            + organization_id UUID FK NOT NULL   (backfill = default org, then SET NOT NULL)
            + workspace_id UUID FK NULL, owner_user_id UUID FK NULL
            CHECK (scope <> 'WORKSPACE' OR workspace_id IS NOT NULL)
            CHECK (scope <> 'PRIVATE_USER' OR owner_user_id IS NOT NULL)
            INDEX (organization_id, scope), INDEX (workspace_id), INDEX (owner_user_id)
source_artifacts.sha256: UNIQUE → UNIQUE(sha256, ?)  — NO. Keep global uniqueness of bytes; a second upload of the same bytes
            by another owner creates a new regulations/regulation_versions row pointing at the same artifact (content-addressed, no leak: bytes are only reachable through an authorized version).
ingestion_jobs(id, version_id FK NOT NULL, status QUEUED|RUNNING|SUCCEEDED|FAILED|QUARANTINED|CANCELLED,
               stage, attempt, max_attempts DEFAULT 3, run_after, locked_at, locked_by,
               error_code, error_public_message, error_internal_ref (= ingestion_runs.id),
               requested_by_user_id, created_at, started_at, completed_at)
               INDEX (status, run_after) — worker poll; UNIQUE partial index (version_id) WHERE status IN ('QUEUED','RUNNING') — one live job per version
```
Downgrade: drop `ingestion_jobs`, drop the four columns/constraints (0004) (scope information on uploaded documents is lost on downgrade — acceptable, documented).

### `0003_conversations` (shipped in Phase C)
```
conversations(id, user_id FK, organization_id FK, workspace_id FK NULL, title, title_locked BOOL DEFAULT false,
              source_scope JSONB NOT NULL, summary TEXT NULL, created_at, updated_at, archived_at)
              INDEX (user_id, updated_at DESC)
messages(id, conversation_id FK ON DELETE CASCADE, ordinal INT, role user|assistant, content TEXT, answer_mode NULL,
         model NULL, provider NULL, trace_id NULL → query_traces.trace_id (no FK; traces may be pruned), warnings JSONB, created_at)
         UNIQUE (conversation_id, ordinal)   -- ordinal is the ordering key; created_at ties inside one transaction
message_citations(id, message_id FK CASCADE, chunk_id FK ON DELETE SET NULL, version_id FK ON DELETE SET NULL,
         citation_order, citation_label TEXT, quote_excerpt TEXT NULL, retrieval_rank INT NULL, scope TEXT)
```
Citations keep `citation_label`/`scope` denormalised so a re-ingested (cascaded) chunk does not erase history; the UI shows "source version no longer active" when `chunk_id IS NULL`.

Tests build the schema with `alembic upgrade head` (`tests/conftest.py`); a new `tests/integration/test_migrations.py` runs head → `0001` → head on a DB holding corpus rows so every downgrade is exercised in CI, not by hand.

## Decision 4 — Authorization predicate (one definition, two mirrors, one equivalence test)

`ScopeFilter` gains an `authz: Authz` value built once per request from the principal's memberships:

```
Authz(user_id, organization_ids, workspace_ids, source_scopes ⊆ {AUTHORITATIVE_ORG, WORKSPACE, PRIVATE_USER}, document_ids?)
```

Predicate (document-level, evaluated before any scoring):
```
(scope = AUTHORITATIVE_ORG ∧ organization_id ∈ organization_ids)
∨ (scope = WORKSPACE      ∧ workspace_id ∈ workspace_ids)
∨ (scope = PRIVATE_USER   ∧ owner_user_id = user_id)
∧ scope ∈ source_scopes            -- the user's *selected* subset of what they may see
∧ (document_ids ⊆ visible)          -- optional "ask this document"
```
Applied in `retrieval/base.scoped_statement` (SQL, dense + exact legs) and mirrored in `retrieval/sparse.VersionMeta.in_scope` (BM25 pre-filter, which already runs before scoring). `tests/unit/test_scope_equivalence.py` drives both with the same table of (document, authz) cases and asserts identical answers — the guard against the two drifting.

Requested source scope on `/ask` and conversations is validated against membership: requesting a workspace you are not in → 403; nothing is silently narrowed.

API keys keep today's role model for scripts/CI. A browser session resolves to a persisted `User`; `Principal` gains `user_id`, `organization_ids`, `workspace_ids`, `roles`. Role → scope map extended with the four TRD roles (`engineer` = today's `Engineer` + `document:upload`; `knowledge_admin` = `DataIngestor` + `document:promote`; `auditor` = `Auditor`; `org_admin` = `SafetyAdmin`). Legacy role names remain valid for API keys.

## Decision 5 — Async ingestion: DB-backed queue + worker process (D-003)

`documents.enqueue(session, version_id)` inserts an `ingestion_jobs` row. `safety-assistant worker` loops: `SELECT … FROM ingestion_jobs WHERE status='QUEUED' AND run_after<=now() ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1`, marks RUNNING, runs `ingest_version(session, version_id, job)` (a refactor of `ingest_source` that starts from an existing `SourceArtifact` instead of the registry fetch — the stage functions already take bytes + a version), records the `ingestion_runs` row as `error_internal_ref`, sets SUCCEEDED / FAILED (retry with backoff `2^attempt` min until `max_attempts`) / QUARANTINED (terminal). Public error text comes from a fixed `error_code → message` table; internal detail stays in `ingestion_runs.error`. Tests call `worker.run_once(session)` synchronously. Registry ingest (`POST /admin/ingest`) becomes an enqueue as well so there is one execution path.

`ponytail:` single-table polling queue; upgrade path is Celery/Redis behind the same `enqueue()`/`run_once()` seam if job throughput or multi-region needs it.

## Decision 6 — API contract (all routes already live under `/api/v1`; new ones join them)

```
GET   /v1/me                                   user, memberships, workspaces, preferences
PATCH /v1/me/preferences
POST  /v1/auth/dev-login  (dev only)           sets HttpOnly session cookie   (D-007)
POST  /v1/auth/logout

POST  /v1/documents            multipart: file + title, document_type, scope, workspace_id?, version_label?, effective_from?, notes?
                               → {document_id, document_version_id, ingestion_job_id, status: "UPLOADED"}   (duplicate sha for same owner/scope → 200 with existing ids)
GET   /v1/documents            ?scope&status&document_type&workspace_id&q   (authorized predicate in SQL)
GET   /v1/documents/{id}       metadata + versions + latest job
POST  /v1/documents/{id}/archive
POST  /v1/documents/{id}/promote            knowledge_admin/org_admin; → AUTHORITATIVE_ORG; audit event
GET   /v1/ingestion-jobs/{id}
POST  /v1/ingestion-jobs/{id}/retry         owner or admin; only FAILED; QUARANTINED never retried

POST  /v1/conversations        {title?, workspace_id?, source_scope}
GET   /v1/conversations        ?q&archived&cursor
GET   /v1/conversations/{id}   messages + citations + source_scope
PATCH /v1/conversations/{id}   {title? (sets title_locked), archived?, source_scope?}
POST  /v1/conversations/{id}/messages   {content, as_of?, regulation_keys?} → runs /ask pipeline under the conversation's source_scope, persists user+assistant messages + citations, returns AnswerResponse + message ids
```
Existing `/ask`, `/search`, `/evidence/{id}`, `/regulations*`, `/admin/*`, `/health/*` are untouched (tests and the retrieval eval depend on them); `/ask` additionally accepts `source_scope`. Moving them under `/v1` is a Phase G decision.

Browser auth: HS256 JWT (pyjwt already a dependency) in an `HttpOnly; SameSite=Lax; Secure` cookie, 12 h; cookie-authenticated mutating requests must carry `X-Requested-With: safety-assistant` (CSRF). `APP_ENV=production` refuses `dev-login`.

## Decision 7 — Conversation context is wording context, not evidence

`POST /v1/conversations/{id}/messages` passes the last N turns (and `summary` when set) to generation as `prior_turns`; the prompt labels it "conversation context — not a source". Citations can only reference evidence retrieved in the current request (already enforced by `generation/citations.py`). No graph change (D-005).

## Consequences

- Cross-user isolation is enforced by one SQL predicate used by every list/detail/retrieval query; the isolation test matrix in `docs/product/06_SECURITY_PRIVACY_AND_MEMORY.md` becomes `tests/security/test_isolation.py` (two users, two workspaces, two orgs).
- Retrieval regression must be re-run after `0003` (new joins/columns in the scope statement; expected metric delta: none).
- New services: none. New backend dependency: `python-multipart` (FastAPI's only multipart parser; not in `uv.lock` today). Frontend adds shadcn/ui primitives and zod (TRD-approved).
- Rollback: each migration downgrades; `0003` downgrade loses upload scope metadata.
