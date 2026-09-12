# Decisions — Safety Assistant v2 Rebuild

One entry per decision. Status: `proposed` (needs owner confirmation), `decided`, `superseded`.
Decisions that touch an approval condition in `CLAUDE.md` stay `proposed` until the owner confirms.

## D-001 — Baseline recovery points (decided, 2026-09-12)
Tag `pre-v2-product-rebuild` and branch `backup/pre-v2-product-rebuild` at `0c7afd1`. The earlier `pre-rebuild-baseline` (af051f2) is kept. No history rewrite, no force-push.

## D-002 — Forward migrations only (decided)
`migrations/versions/0001` stays. Identity/workspace, document scope/jobs and conversations arrive as `0002+`. Squash requires explicit owner approval (`CLAUDE.md` approval conditions) — not requested.

## D-003 — Async ingestion transport (decided by owner 2026-09-12: A, PostgreSQL-backed job table + worker process)
TRD lists "existing proven Celery/worker abstraction" and Redis; **neither exists in the repo** (`grep` for celery/redis in `src`: only a rate-limiter comment). Options:
- **A (default): `ingestion_jobs` table + `SELECT … FOR UPDATE SKIP LOCKED` worker (`safety-assistant worker` CLI).** Zero new services, transactional with the version row, visible via the same DB the UI already reads; satisfies "async", "bounded retries", "idempotent". Ceiling: single-DB polling; upgrade path is Celery/Redis behind the same `enqueue()` seam when throughput demands.
- B: Add Redis + Celery now. New service + dependency without a measured need → violates "minimal dependencies" and the "no major framework/service not in TRD without decision" gate (Redis *is* in TRD, but the abstraction it references does not exist).
Proceeding with A; will switch to B if the owner prefers.

## D-004 — Mapping existing axes to v2 source scope (proposed → default below)
Existing: `regulations.authority_level` (AUTHORITATIVE…SYNTHETIC) and `regulations.data_class` (PUBLIC/CONFIDENTIAL) narrow retrieval in `ScopeFilter`.
v2 adds `scope ∈ {AUTHORITATIVE_ORG, WORKSPACE, PRIVATE_USER}` + `organization_id`/`workspace_id`/`owner_user_id` on the logical document (`regulations` table = logical Document; renaming the table is not worth a data migration — a `Document` alias/domain name is used in new code and docs).
Rule: registry-ingested sources → `AUTHORITATIVE_ORG` (org = the single seeded organization). Uploads → `PRIVATE_USER` (default) or `WORKSPACE` (explicit choice, requires membership). `data_class` stays for provider routing policy; `authority_level` stays for the relevance guard. Authorization predicate = scope ∧ (org membership | workspace membership | owner) evaluated in SQL before ranking.

## D-005 — Keep the bounded LangGraph agent for `/ask`; do not extend it (decided for Phases B–F, revisit in G)
It is tested, deterministic in routing, and inside budget. Conversation context (prior turns) is passed as *wording context only* to generation, never as evidence — implemented in `generation/`, not in the graph. If wiring conversation context requires touching graph state in more than one place, replace the graph with plain functions (ADR required).

## D-006 — Frontend design defaults pending worksheet (proposed)
`11_DESIGN_DECISION_WORKSHEET.md` is unfilled. Defaults taken from `03_UI_UX_DESIGN_SPEC.md`:
engineering workstation · three-pane desktop with collapsible evidence panel, drawer ≤1279px · Engineering Cobalt palette · nav Chat/Documents/Uploads/Settings/Admin(role) · upload default scope **private user** · mobile = basic chat/evidence/status · shadcn/ui + Tailwind 4, Inter.
No Figma stage is available in this environment; low-fi approval happens via the route/component inventory in Phase B before hi-fi code.

## D-007 — Development auth (proposed → default)
Keep `auth_mode=none|api_key|oidc`. Add a dev-only `dev_login` flow that maps to seeded users (engineer / knowledge_admin / auditor / second engineer for isolation tests). Production settings already refuse `none`; they will also refuse dev login. Browser session = HttpOnly cookie carrying a signed session token; API keys remain for scripts/CI.

## D-008 — Root spec package location (decided)
Specs stay at repo root during the rebuild (CLAUDE.md references them by root path). Moved to `docs/product/` in Phase G with CLAUDE.md updated in the same commit.

## D-009 — Lifecycle vocabulary (decided, ADR-0029 §2)
Internal `VersionStatus` unchanged; spec statuses (`UPLOADED…READY`) are a pure API/UI mapping. `READY` ≡ `ACTIVE`.

## D-010 — Roles (decided, ADR-0029 §4)
Membership roles use TRD names (`engineer`, `knowledge_admin`, `auditor`, `org_admin`) mapped onto the existing scope model; legacy role names stay valid for API keys so the 30 security tests and CI keys keep working.

## D-011 — API prefix (decided, corrected in Phase C)
Every existing route already lives under `/api/v1`; new product routes join the same prefix. No split, nothing to move in Phase G.

## D-012 — Browser session (decided by owner 2026-09-12; ADR-0029 §6)
HS256 JWT (pyjwt) in `HttpOnly; SameSite=Lax; Secure` cookie, 12 h; mutating cookie-auth requests require `X-Requested-With`. `dev-login` refused when `APP_ENV=production`. OIDC remains the production path (login redirect → cookie).

## D-013 — Frontend route/component inventory (Phase B low-fi gate; defaults from `03_UI_UX_DESIGN_SPEC.md`)
Routes: `/login`, `/app/home`, `/app/chat`, `/app/chat/[id]`, `/app/documents`, `/app/documents/upload`, `/app/documents/[id]`, `/app/ingestion`, `/app/settings`, `/app/admin/{corpus,ingestion,audit}` (role-gated).
Shell: `AppShell` (top bar: WorkspaceSwitcher · SourceScopeSelector · SystemStatus · user menu; left nav 240px; evidence panel 400px collapsible, sheet <1280px).
Feature components: chat = `ConversationList`, `ConversationHeader`, `MessageList`, `AssistantMessage` (answer-mode badge Grounded / Evidence only / Insufficient evidence), `CitationMarker`, `Composer`; evidence = `EvidencePanel`, `EvidenceCard`; documents = `DocumentTable`, `DocumentStatus`, `UploadDropzone`, `UploadMetadataForm`, `IngestionTimeline`; common = `EmptyState`, `ErrorState`, `PermissionDenied`, `LoadingSkeleton`.
Primitives: shadcn/ui (Button, Input, Textarea, Select, Checkbox, Dialog, DropdownMenu, Tooltip, Tabs, Sheet, Table, Badge, Progress, Skeleton, Toast, Alert). Tokens: Engineering Cobalt as CSS variables in `styles/tokens.css`; Inter; 4 px spacing; radius 8/12.
Server state: TanStack Query for documents/jobs polling and conversations; forms: React Hook Form + zod for upload metadata only. No global client-state library.

## D-014 — Migration numbering (decided, Phase C)
`0002_identity`, `0003_conversations` shipped in Phase C; document scope + ingestion jobs become `0004` in Phase D (ADR-0029 text updated).

## D-015 — API-key principals own no user data (decided, Phase C)
API keys remain role-only (scripts/CI/eval). `/me`, conversations and uploads require a persisted user (`require_user` → 403). OIDC subjects are resolved to a user row only when one exists (provisioned via `safety-assistant users add`); no just-in-time membership.
