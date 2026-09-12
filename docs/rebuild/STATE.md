# Safety Assistant v2 — Rebuild State

Compact progress file. Update at every phase boundary. Do not narrate; record facts.

## Baseline (Phase A — 2026-09-12)

| Item | Value |
|---|---|
| Starting SHA | `0c7afd1096a54aafe2e3f6e2574a5b611176c0d4` |
| Tag / backup branch | `pre-v2-product-rebuild` / `backup/pre-v2-product-rebuild` (both at 0c7afd1) |
| Older recovery tag | `pre-rebuild-baseline` (af051f2, pre-v1-rebuild CAE product) |
| Uncommitted at start | `CLAUDE.md` rewritten (137 lines, v2 contract); untracked spec package `00_…11_*.md`, `AGENTS.md`, `README_PACKAGE.md`, `.claude/skills/*` |
| Prior rebuild ledger | `docs/rebuild-ledger.md` (v1 rebuild M0–M17, measured results) — historical; superseded by this directory |

### Baseline gates (run locally 2026-09-12, HEAD 0c7afd1)

| Gate | Result |
|---|---|
| `ruff check src tests scripts migrations` | pass |
| `ruff format --check` | pass (129 files) |
| `mypy --strict src scripts/eval scripts/maintenance` | pass, 100 files |
| `pytest` (unit 53, parser_golden 9, security 30, integration 16, e2e 5, evaluation 3, retrieval_regression 3) | **119 passed** in 74 s (Postgres on :5433 running) |
| Frontend `tsc --noEmit`, `eslint .` | pass |
| Retrieval baseline (`evals/results/retrieval_regulatory_v1_latest.json`, git f82ce5a, 47 cases / 39 section-truth, 21,910 chunks) | full leg R@5 0.759 · R@10 0.901 · MRR 0.636 · nDCG@10 0.671; regression floor MRR ≥ 0.60, 23/23 stable cases |
| Playwright | 3/3 at f82ce5a (not re-run this phase) |
| Docker build + smoke | executed at 0c7afd1 (see rebuild-ledger §9); not re-run this phase |

**Retrieval baseline to preserve:** MRR 0.636 / R@10 0.901 on `regulatory_v1` full leg. Any retrieval-affecting change re-runs `scripts/eval/retrieval.py` and `tests/retrieval_regression`.

## Current architecture (as found)

Backend `src/safety_assistant/` — 7,507 LOC, FastAPI + SQLAlchemy 2 + Alembic (1 migration) + pgvector, fastembed all-MiniLM-L6-v2 (384-d, HNSW), rank-bm25 (cached in-memory), LangGraph bounded agent, OTel + Prometheus.

| Area | State | Gap vs v2 spec |
|---|---|---|
| Identity | `api/dependencies/auth.py`: modes none/api_key/oidc → `Principal(subject, role, scopes)`; 5 roles fixed in code | no `users`/`organizations`/`memberships`/`workspaces` tables; no stable `user_id`; roles ≠ TRD roles |
| Corpus model | `regulations` → `regulation_versions` (lifecycle DISCOVERED…ACTIVE/SUPERSEDED/QUARANTINED/FAILED) → `sections`/`chunks`/`chunk_embeddings`; `source_artifacts` content-addressed | no `Document`/`DocumentVersion` scope (`AUTHORITATIVE_ORG/WORKSPACE/PRIVATE_USER`); `data_class` PUBLIC/CONFIDENTIAL is the only isolation axis |
| Ingestion | `ingestion/workflows/ingest.py` — registry-allowlisted sources only, idempotent, quarantine, atomic activation; **runs synchronously** inside `POST /admin/ingest` via `run_in_threadpool` | no user upload endpoint; no queue/worker; no Redis; no `ingestion_jobs` per-version job with public error |
| Retrieval | `retrieval/service.py` — SQL scope (status/temporal/regulation/data_class) before dense+sparse+exact → RRF → rerank → guard → parent/xref expansion | scope predicate must grow to owner/workspace/org |
| Generation | `generation/` — GroundedDraft, evidence gate, citation + numeric validation, modes GENERATED/EVIDENCE_ONLY/ABSTAINED, `query_traces` | none for MVP |
| Conversations / memory | none | `conversations`, `messages`, `message_citations`, `user_preferences` all missing |
| API | `/search`, `/ask`, `/evidence/{id}`, `/regulations`, `/regulations/{key}/versions|diff`, `/feedback`, `/admin/*`, `/health/*` | `/v1/documents*`, `/v1/ingestion-jobs*`, `/v1/conversations*`, `/v1/me` missing |
| Frontend | `frontend/` Next.js 16 / React 19 / Tailwind 4: single page (AuthPanel + ChatPanel + CitationPanel), 3 Playwright specs, ~650 LOC | whole IA (login/home/chat/documents/upload/detail/ingestion/settings/admin) to build; no shadcn, no design tokens |
| Infra | Dockerfile (non-root), compose (postgres only at root; api+postgres+minio in `infra/docker`), Terraform (unapplied), 4 GitHub workflows | needs worker service + Redis (or DB-backed queue — see DECISIONS) |

## Phase log

| Phase | Status | Evidence |
|---|---|---|
| A baseline + inventory | **done 2026-09-12** | this file; `FILE_LEDGER.md`; `DECISIONS.md` D-001…D-006 |
| B architecture / schema plan | **done 2026-09-12** | `docs/ADR/0029-v2-product-domain-schema-and-authorization.md`; DECISIONS D-009…D-013 |
| C backend product foundation (identity, workspace, conversations, preferences) | **done 2026-09-12** | migrations `0002_identity`, `0003_conversations`; `identity/service.py`, `conversations/service.py`; routes `me.py` (`/me`, `/me/preferences`, `/auth/dev-login`, `/auth/logout`), `conversations.py`; `Principal` carries user/org/workspace ids; cookie sessions + CSRF header; conversation context → `<conversation_context>` (prompt `grounded_v2`); CLI `users add`; tests: unit 60, security 34, integration 24 (incl. migration head→0001→head) — **135 passed**; lint/format/mypy clean |
| D document upload + async ingestion | next | migration `0004_document_scope_and_jobs`; `/api/v1/documents*`, `/api/v1/ingestion-jobs*`; `safety-assistant worker`; authz predicate in `scoped_statement` + BM25 mirror + equivalence test |
| E frontend rebuild | pending (design gate: `11_DESIGN_DECISION_WORKSHEET.md` unfilled → defaults in DECISIONS D-006) | |
| F tests / security / eval | pending | |
| G cleanup | pending | |
| H final verification + report | pending | |

## Owner confirmations

- D-003, D-012: confirmed 2026-09-12.
- D-006/D-013 frontend defaults: still open; needed before Phase E hi-fi.
