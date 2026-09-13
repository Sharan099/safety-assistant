# File Ledger — Safety Assistant v2

Classification of every tracked top-level path at baseline `0c7afd1` (tag `pre-v2-product-rebuild`).

Legend: **KEEP** (unchanged), **MIGRATE** (move/extend, behaviour preserved by existing tests), **REWRITE** (replace; new tests), **DELETE** (remove after replacement verified; git history is the archive).

No path may be deleted unless it is listed DELETE here with its replacement present and tests green.

## Backend — `src/safety_assistant/`

| Path | Action | Reason | Replacement / target |
|---|---|---|---|
| `retrieval/*` (service, sparse, dense, fusion, rerank, filters, context, expansion, base) | MIGRATE | benchmarked hybrid retrieval (MRR 0.636 baseline) | same package; `ScopeFilter` gains owner/workspace/org predicate; regression tests must stay green |
| `generation/*` (service, grounding, citations, schemas, prompts) | KEEP | evidence gate + citation validation are contract-tested | unchanged; `AnswerResponse` reused by message persistence |
| `ingestion/workflows/ingest.py` | MIGRATE | proven lifecycle/idempotency/quarantine/atomic activation | split into stage functions callable from a worker; add upload-sourced entry (bytes from blob store, not registry) |
| `ingestion/{parse,normalize,chunk,index,validation,diff}` | KEEP | golden/e2e tested | unchanged |
| `ingestion/fetch/{blobstore,s3}.py` | KEEP | object-storage abstraction already exists | upload path writes through `BlobStore.put` |
| `ingestion/fetch/http.py` | KEEP | SSRF-safe fetcher | unchanged |
| `ingestion/sources/registry.py` | KEEP | authoritative-corpus allowlist | remains the only path to `AUTHORITATIVE_ORG` besides admin promotion |
| `persistence/models/regulatory.py` | MIGRATE | canonical corpus model | add `scope`, `owner_user_id`, `workspace_id`, `organization_id` on `regulations` (= logical Document); versions keep lifecycle; new tables in new migrations |
| `persistence/models/operations.py` | MIGRATE | runs/events/traces/feedback | add `ingestion_jobs` (per-version job, public error, attempt) or extend `ingestion_runs` — decide in Phase B |
| `persistence/{base,session}.py` | KEEP | | |
| `api/dependencies/auth.py`, `oidc.py` | MIGRATE | api_key/OIDC verification + constant-time compare proven | resolve `Principal` → persisted `User` + memberships; map roles to TRD roles (engineer/knowledge_admin/auditor/org_admin) |
| `api/routes/query.py` | MIGRATE | `/ask` & `/search` contract used by tests | `/ask` gains conversation_id + source_scope; response persisted as message + citations |
| `api/routes/admin.py` | MIGRATE | registry ingest + runs/events/traces | ingest becomes enqueue, not inline; add promotion endpoint |
| `api/routes/{health,versions}.py` | KEEP | | |
| `api/middleware/*` | KEEP | request id, rate limit, security headers | |
| `api/app.py` (or main) | MIGRATE | | mount new routers |
| `agents/*` (LangGraph graph/state/tools, 540 LOC) | MIGRATE → evaluate | bounded, tested; CLAUDE.md forbids *adding* agents, not keeping a bounded deterministic one | keep as-is for `/ask`; do not extend. Candidate for REWRITE to plain functions in Phase G if it blocks conversation context (D-005) |
| `domain/*` | KEEP | lifecycle state machine, temporal parse | new `domain/documents.py` enum `SourceScope`, `domain/identity.py` |
| `providers/*` | KEEP | production-fake prohibition enforced in settings | |
| `observability/*` | KEEP | | add spans for persistence/upload/job |
| `security/injection.py` | KEEP | | |
| `evaluation/*` | KEEP | | |
| `config/settings.py` | MIGRATE | | add upload bounds, queue settings, session/cookie settings |
| `cli.py` | MIGRATE | | add `worker` command |
| **new** `identity/service.py`, `conversations/service.py`, `api/routes/{me,conversations}.py`, `persistence/models/{identity,conversations}.py` | — | v2 product foundation | created Phase C |
| **new** `documents/service.py`, `workers/ingestion.py`, `retrieval/authz.py`, `domain/documents.py`, `api/routes/documents.py` | — | upload + async ingestion + authz predicate | created Phase D |

## Frontend — `frontend/`

| Path | Action | Reason | Replacement |
|---|---|---|---|
| `app/page.tsx`, `app/layout.tsx`, `app/globals.css` | REWRITTEN (Phase E) | single-page proof of concept; v2 IA is `/login`, `/app/*` | App Router route tree per `docs/product/03_UI_UX_DESIGN_SPEC.md` |
| `components/{ChatPanel,AnswerCard,CitationPanel,AuthPanel,SystemStatus}.tsx` | DELETED (Phase E, replacement present) | logic reusable, structure not | `components/{ui,shell,chat,evidence,documents,common}/` |
| `lib/apiClient.ts` → `lib/api.ts`, `lib/types.ts`, `lib/errors.ts` | MIGRATED (Phase E) | typed client + request tracing kept; cookie session + CSRF header | documents/conversations/me endpoints added |
| `hooks/{useAuth,useChat,useHealth}.ts` | DELETED (Phase E) | token-in-hook auth model replaced by session/user model | `features/queries.ts` |
| `tests/ask.spec.ts` + playwright config | REPLACED (Phase E) | superseded by the 5 required flows | `tests/e2e/flows.spec.ts` |
| `package.json`, `tsconfig.json`, `next.config.ts`, eslint config | KEEP | | add shadcn/ui, zod as needed (TRD-approved) |
| `test-results/` | DELETE | gitignored artefact dir; only `.last-run.json` tracked | — |

## Tests — `tests/` (119 passing; none weakened)

| Path | Action | Note |
|---|---|---|
| `unit/`, `parser_golden/`, `security/`, `integration/`, `e2e/`, `evaluation/`, `retrieval_regression/`, `support/`, `conftest.py` | KEEP | add new categories: `api_contract/`, `ingestion_e2e/` (upload path), security isolation tests per `06_*.md` |

## Migrations — `migrations/`

| Path | Action |
|---|---|
| `versions/0001_canonical_regulatory_schema.py`, `env.py`, `alembic.ini` | KEEP — forward migrations only: `0002_identity`, `0003_conversations`, `0004_document_scope_and_jobs` shipped. No squash (D-002) |

## Evals, scripts, data

| Path | Action | Note |
|---|---|---|
| `evals/datasets/regulatory_v1.yaml`, `evals/baselines/*`, `evals/results/*` | KEEP | baseline to preserve |
| `scripts/eval/{retrieval,load_test}.py`, `scripts/maintenance/verify_registry.py` | KEEP | |
| `knowledge/00_registry/sources.yaml` + gitignored PDFs | KEEP | authoritative corpus allowlist |
| `data/` (gitignored artifacts) | KEEP | regenerable |
| `Knowledge source/` (gitignored, 1.4 GB) | KEEP untracked | user's raw folder; never read by code |

## Infra / CI

| Path | Action | Note |
|---|---|---|
| `infra/docker/{Dockerfile,entrypoint.sh,compose.yaml}` | MIGRATE | add worker service; add Redis only if D-003 chooses it |
| `docker-compose.yml` (root, dev postgres) | KEEP | |
| `infra/monitoring/*`, `infra/terraform/*` | KEEP | add worker/queue resources when D-003 settled |
| `.github/workflows/{ci,security,eval,release}.yml` | MIGRATE | add frontend build + Playwright job; migration up/down test |

## Docs

| Path | Action | Note |
|---|---|---|
| `CLAUDE.md` (rewritten), `AGENTS.md`, `.claude/skills/*` | KEEP | v2 contract |
| `00_…11_*.md`, `README_PACKAGE.md` (root spec package) | MIGRATE → `docs/product/` at Phase G | canonical specs; keep at root until rebuild done so paths in CLAUDE.md stay valid, then move and update CLAUDE.md. `05_`, `10_`, `11_` are process docs → `docs/rebuild/` or DELETE after use |
| `docs/rebuild/v1-ledger.md` | MIGRATE → `docs/rebuild/v1-ledger.md` | historical record of the first rebuild; measured numbers referenced by README |
| `docs/{architecture,data-lineage,retrieval-design,evaluation,security-threat-model,operations-runbook,incident-response}.md` | MIGRATE | update for identity/workspace/upload/conversations; do not duplicate spec content |
| `docs/ADR/0001–0028` | KEEP | new ADRs continue at 0029 |
| `README.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`, `Makefile`, `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore` | MIGRATE | update commands/config as features land |

## Deletions executed

| Path | Commit | Replacement / proof |
|---|---|---|
| `frontend/lib/apiClient.ts`, `frontend/hooks/{useAuth,useChat,useHealth}.ts`, `frontend/components/{AnswerCard,AuthPanel,ChatPanel,CitationPanel,SystemStatus}.tsx`, `frontend/tests/ask.spec.ts` | 7126278 | `lib/api.ts`, `features/queries.ts`, `components/{shell,chat,evidence,documents,common}`, `tests/e2e/flows.spec.ts` (5/5 passing) |
| `README_PACKAGE.md` (reading order for the spec package) | Phase G | superseded by `CLAUDE.md` canonical list + `docs/product/` |
| `frontend/test-results/.last-run.json` (tracked artefact) | Phase G | gitignored directory |
| moved: root `00–08_*.md` → `docs/product/`, `09–11_*.md` → `docs/rebuild/`, `docs/rebuild-ledger.md` → `docs/rebuild/v1-ledger.md` | Phase G | references rewritten in CLAUDE.md, skills, ADR-0029, rebuild docs |
