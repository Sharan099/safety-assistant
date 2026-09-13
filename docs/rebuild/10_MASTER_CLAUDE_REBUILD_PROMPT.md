# Master Prompt — Claude Opus 5 Full v2 Rebuild

Use this from the repository root after placing the specification package into the repo.

---

You are the lead engineer for the Safety Assistant v2 rebuild.

Use Claude Opus 5.

Your job is to rebuild the repository into the product described by the canonical specifications while preserving already-proven backend behavior.

## Read first

1. `CLAUDE.md`
2. `00_MASTER_REBUILD_PLAN.md`
3. `01_PRD.md`
4. `02_TRD.md`
5. `03_UI_UX_DESIGN_SPEC.md`
6. `04_APP_FLOWS.md`
7. `06_SECURITY_PRIVACY_AND_MEMORY.md`
8. `07_TESTING_EVALUATION_RELEASE.md`
9. `08_CLEANUP_AND_MIGRATION_PLAN.md`

Do not copy these specs into working notes. Reference paths.

## Existing context

The repository is already a production-oriented regulatory RAG implementation. Preserve verified behavior such as version-aware ingestion, hybrid retrieval, exact-clause retrieval, RRF, reranking, evidence gating, citation validation, authorization filtering, observability, evaluation baselines and hardened deployment unless code inspection proves otherwise.

This is a product rebuild, not permission to replace working engineering with fashionable frameworks.

## Execution style

Run the complete rebuild in one orchestrated session using explicit phase gates.

At discovery use no more than three concurrent read-only subagents:
- frontend/product;
- backend/data/security;
- tests/cleanup.

Default subagent depth: one.

Use project Skills when relevant.

Maintain:
- `docs/rebuild/STATE.md`
- `docs/rebuild/FILE_LEDGER.md`
- `docs/rebuild/DECISIONS.md`

### Phase 0 — baseline

Before editing:
- verify git status;
- record HEAD;
- verify/create `pre-v2-product-rebuild` recovery tag if safe and absent;
- run relevant backend/frontend tests;
- capture retrieval/evaluation baseline;
- identify the current running architecture from code, not only docs.

If baseline tests fail, record failures before changing anything.

### Phase 1 — inventory

Create FILE_LEDGER.md.

Classify major tracked areas:
- KEEP
- MIGRATE
- REWRITE
- DELETE

Do not delete yet.

Map data model, auth, ingestion, retrieval, generation, observability, frontend, tests and deployment.

### Phase 2 — architecture/design plan

Compare actual implementation against PRD/TRD/UI specs.

Write only required deviations/decisions into DECISIONS.md.

Do not redesign already-correct retrieval merely to make code new.

Prepare:
- schema delta;
- API delta;
- frontend route/component plan;
- migration strategy;
- cleanup plan.

### Phase 3 — backend product foundation

Implement/verify:
- user/org/workspace/membership;
- conversations/messages/citations;
- preferences;
- document scopes;
- document versions;
- ingestion jobs;
- authorization-scoped services;
- typed API contracts.

Use migrations and add tests.

### Phase 4 — upload + ingestion

Implement:
- upload/presigned flow appropriate to current architecture;
- object storage;
- metadata;
- job creation/status;
- validated parser/chunker/embed/index pipeline;
- READY-only activation;
- failure/quarantine.

Private/workspace documents must never leak across users/workspaces.

### Phase 5 — conversation/history

Implement:
- create/list/open/rename/archive;
- persistent messages;
- citation persistence;
- source scope;
- restore after login;
- explicit preferences.

Do not create hidden factual regulatory memory.

### Phase 6 — frontend clean rebuild

Rebuild using `03_UI_UX_DESIGN_SPEC.md`.

Use a project-owned shadcn-based component system and consistent tokens.

Required routes:
- login;
- home;
- chat/[id];
- documents;
- documents/upload;
- documents/[id];
- ingestion;
- settings;
- role-gated admin where supported.

Required layout:
- left navigation;
- main conversation/work area;
- collapsible evidence panel.

Implement:
- source scope selector;
- persistent history;
- evidence cards;
- authoritative/private distinction;
- document table;
- upload flow;
- ingestion timeline;
- empty/loading/error/permission states.

Visually verify at desktop, tablet and a narrow basic viewport.

### Phase 7 — tests

Required:
- backend unit;
- integration;
- API contracts;
- security isolation;
- ingestion E2E;
- retrieval regression;
- frontend checks;
- Playwright.

Critical E2E:
1. login → ask → open evidence;
2. login → upload → READY → ask uploaded doc;
3. logout/login → restore conversation;
4. cross-user private document denial;
5. ingestion failure UI.

### Phase 8 — cleanup

Only now execute ledger-approved DELETE actions.

Remove superseded frontend, dead scripts, stale product docs, duplicate harnesses, obsolete fixtures, unused dependencies, logs/generated junk.

Git history is the archive. Do not leave old implementations in active folders.

Re-run reference/import search.

### Phase 9 — production checks

Run applicable:
- lint;
- type check;
- full relevant tests;
- retrieval eval;
- frontend production build;
- Playwright;
- available security scans;
- Docker build/smoke;
- migration from clean DB;
- representative upgrade migration if required.

Do not claim unavailable checks passed.

### Phase 10 — docs

Rewrite README for v2:
1. problem;
2. product flow/screens;
3. measured results;
4. architecture;
5. document ingestion;
6. evidence/citation model;
7. security/privacy;
8. local run;
9. evaluation;
10. known limitations.

Keep active docs current only.

### Phase 11 — final report

Return verified facts only:
- starting/final SHA;
- final tree;
- deleted paths;
- preserved/migrated capabilities;
- schema changes;
- endpoints;
- UI routes;
- tests by category;
- retrieval metrics vs baseline;
- E2E result;
- Docker result;
- security checks;
- known limitations;
- exact run commands.

## Rules

- Do not ask for approval on ordinary implementation choices already fixed by specs.
- Stop for approval conditions defined in CLAUDE.md.
- Never weaken tests/evals.
- Never hard-code evaluation answers.
- Never use fake production dependencies.
- Never retrieve unauthorized chunks then filter afterward.
- Never auto-promote user uploads to authoritative data.
- Never treat chat history as evidence.
- Never claim a check passed if it was not run.
- Prefer simple testable code.
- After verification, prefer deletion over competing obsolete implementations.

Begin with Phase 0.
