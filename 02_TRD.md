# Technical Requirements Document — Safety Assistant v2

## Architecture principles

1. Deterministic before probabilistic where rules can guarantee safety.
2. Authorization before ranking.
3. Evidence before generation.
4. Versioned data lineage.
5. Async for expensive ingestion.
6. No silent fake dependencies in production.
7. Conversation memory is not knowledge truth.
8. Small replaceable modules over framework entanglement.

## Logical architecture

```text
                         ┌─────────────────────────┐
                         │       Next.js UI        │
                         │ Engineering Workbench   │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │             FastAPI               │
                    │ Auth / Policy / API / Streaming   │
                    └───────┬──────────────┬────────────┘
                            │              │
                  ┌─────────▼──────┐  ┌────▼───────────┐
                  │ Conversations  │  │ Document API   │
                  │ + Preferences  │  │ + Job status   │
                  └─────────┬──────┘  └────┬───────────┘
                            │              │
                 ┌──────────▼──────────────▼──────────┐
                 │        PostgreSQL + pgvector      │
                 │ users/workspaces/docs/chunks/chat │
                 └───────┬─────────────────────┬─────┘
                         │                     │
                ┌────────▼────────┐   ┌────────▼─────────┐
                │ Retrieval       │   │ Queue / Workers  │
                │ auth filters    │   │ ingestion        │
                │ dense/sparse    │   └────────┬─────────┘
                │ exact/RRF       │            │
                │ rerank/expand   │    ┌───────▼────────┐
                └───────┬─────────┘    │ Object Storage │
                        │              └────────────────┘
                ┌───────▼────────┐
                │ Evidence Gate  │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │ LLM Gateway    │
                │ citation rules │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │ answer + cites │
                └────────────────┘
```

## Recommended stack

### Backend
- Python;
- FastAPI;
- Pydantic;
- SQLAlchemy;
- Alembic;
- PostgreSQL 16 + pgvector;
- Redis;
- existing proven Celery/worker abstraction;
- S3-compatible object storage (MinIO local, managed object storage in cloud);
- OpenTelemetry;
- Prometheus.

Do not replace a working component without a measured reason.

### Frontend
- Next.js App Router;
- TypeScript strict mode;
- Tailwind CSS;
- shadcn/ui;
- supported headless primitive layer;
- TanStack Query only where server-state behavior benefits;
- React Hook Form + Zod where useful;
- Playwright;
- Vitest/RTL where useful.

Do not add a global client-state library without a real need.

## Core domain model

### User
```text
id UUID
email
display_name
status
created_at
last_login_at
```

### Organization
```text
id UUID
name
created_at
```

### Membership
```text
user_id
organization_id
role
```

Roles:
```text
engineer
knowledge_admin
auditor
org_admin
```

### Workspace
```text
id UUID
organization_id
name
created_by
created_at
archived_at
```

### WorkspaceMembership
```text
workspace_id
user_id
role
```

### Document
Logical identity.
```text
id UUID
organization_id nullable
workspace_id nullable
owner_user_id nullable
scope enum
canonical_title
document_type
created_at
```

### DocumentVersion
```text
id UUID
document_id
version_label
publication_date nullable
effective_from nullable
effective_to nullable
source_uri nullable
object_key
sha256
mime_type
page_count nullable
status
parser_version
chunker_version
embedding_model
created_at
activated_at nullable
```

### IngestionJob
```text
id UUID
document_version_id
stage
status
attempt
progress_percent nullable
error_code nullable
error_public_message nullable
error_internal_ref nullable
created_at
started_at
completed_at
```

### Chunk
```text
id UUID
document_version_id
ordinal
text
embedding
clause_id nullable
section_path
page_start
page_end
chunk_type
metadata jsonb
```

### Conversation
```text
id UUID
user_id
workspace_id nullable
title
source_scope jsonb
created_at
updated_at
archived_at
```

### Message
```text
id UUID
conversation_id
role
content
answer_mode nullable
model nullable
provider nullable
request_id nullable
created_at
```

### MessageCitation
```text
id UUID
message_id
chunk_id
document_version_id
citation_order
quote_excerpt nullable
retrieval_rank nullable
```

### UserPreference
```text
user_id
default_workspace_id nullable
answer_density
preferred_language
ui_theme
updated_at
```

### AuditEvent
```text
id
actor_user_id nullable
organization_id nullable
action
resource_type
resource_id
request_id
metadata
created_at
```

## Authorization model

Authorization happens before retrieval ranking.

```text
request
→ authenticate
→ resolve organization/workspace
→ build authorized source predicate
→ validate requested filters
→ candidate retrieval
→ ranking
```

Never retrieve globally and remove unauthorized chunks only after ranking.

## Source scopes

```text
AUTHORITATIVE_ORG
WORKSPACE
PRIVATE_USER
```

Every candidate carries identity/scope/version validity metadata.

## Upload APIs

### POST `/v1/documents`
Return:
```json
{
  "document_id": "...",
  "document_version_id": "...",
  "ingestion_job_id": "...",
  "status": "UPLOADED"
}
```

For large files, prefer a presigned object-storage upload pattern.

### GET `/v1/documents`
Filter by workspace/status/scope/type/date.

### GET `/v1/documents/{id}`
Authorized metadata + latest version + ingestion state.

### GET `/v1/ingestion-jobs/{id}`
Stage/progress/error.

### POST `/v1/ingestion-jobs/{id}/retry`
Privileged/safe only.

## Ingestion pipeline

```text
UPLOADED
  ↓
VALIDATING
  ├── invalid → QUARANTINED / FAILED
  ↓
PARSING
  ↓
STRUCTURE EXTRACTION
  ↓
CHUNKING
  ↓
EMBEDDING
  ↓
INDEXING
  ↓
VERIFYING
  ├── QA fail → FAILED
  ↓
READY
```

Required:
- idempotency;
- checksum duplicate detection;
- bounded retries;
- timeout;
- version lineage;
- no partial retrieval availability;
- atomic activation/supersession;
- test fakes isolated from production.

## Conversation APIs

```text
POST /v1/conversations
GET  /v1/conversations
GET  /v1/conversations/{id}
PATCH /v1/conversations/{id}
POST /v1/conversations/{id}/messages
```

Message request can include authorized source-scope selection.

## Memory requirements

Do not build hidden semantic user memory before product evidence justifies it.

MVP:
```text
conversation persistence
+ explicit user preferences
+ optional conversation summary for context compression
```

Conversation summary:
- never cited;
- never overrides evidence;
- scoped to conversation.

## Cache policy

Cache identity must account for source scope, corpus/index version, retrieval config, model/prompt version where relevant, and user/workspace boundaries.

Default:
```text
org-safe deterministic retrieval → bounded cache allowed
private/workspace generated answer → no shared answer cache
```

## Observability

Trace:
```text
HTTP request
├─ auth
├─ policy
├─ dense
├─ sparse
├─ exact-clause
├─ fusion
├─ rerank
├─ expansion
├─ evidence gate
├─ generation
└─ persistence
```

Metrics:
- request latency;
- retrieval/rerank/LLM latency;
- answer mode;
- no-hit/fallback;
- token/cost;
- ingestion duration/failure;
- queue depth;
- freshness lag.

## Security requirements

- OIDC in production;
- secure session/token policy;
- RBAC;
- backend authorization;
- file byte/page bounds;
- magic/MIME validation;
- safe object keys;
- optional malware scanning boundary;
- SSRF-safe remote acquisition;
- rate limiting;
- CSRF strategy for cookie auth;
- audit privileged operations;
- no secrets in client;
- no raw confidential content in telemetry by default;
- cross-user leakage tests;
- indirect prompt-injection tests.

## Migration strategy

1. freeze baseline metrics;
2. refactor around domain boundaries;
3. preserve compatibility tests;
4. rebuild frontend separately;
5. add user/workspace/document/conversation schema;
6. rerun retrieval regression after storage/retrieval changes.

Only reset/squash migrations if there is no production data requirement and the owner explicitly approves a clean baseline.
