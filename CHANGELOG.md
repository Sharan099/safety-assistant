# Changelog

## 0.3.0 — 2026-09-13 (v2: product workspace)

Rebuild into the passive-safety regulatory intelligence workspace (ADR-0029).

- Identity and tenancy: organizations, users, memberships (engineer / knowledge_admin / auditor / org_admin), workspaces, preferences, audit events; HttpOnly session cookies with CSRF header; dev login (never in production); `safety-assistant users add`. Migrations `0002_identity`, `0003_conversations`, `0004_document_scope_and_jobs` (forward only; round-trip tested).
- Persistent conversations with citations and source scope; prior turns reach the model only as `<conversation_context>` data (prompt `grounded_v2`).
- Document workspace: bounded PDF upload (private or workspace scope), same-owner dedupe, PostgreSQL-backed ingestion queue + `safety-assistant worker` (SKIP LOCKED, retries with backoff, public error codes, quarantine terminal), archive, audited promotion to the verified corpus; `POST /admin/ingest` enqueues.
- Authorization predicate (`retrieval/authz.py`) evaluated in SQL before ranking and mirrored in the BM25 pre-filter, with an equivalence test; no identity = authoritative sources only.
- Retrieval: RRF dense weight 0.75; cross-encoder reranker over the top-12 candidates is the default (v1 MRR 0.636 → 0.756, v2 0.709 → 0.808); numeric validator accepts numbers from evidence attributes.
- Evaluation: `regulatory_v2` (262 cases: 47 reviewed + 200 AUTO_GROUNDED + 15 hand-written), grounded case generator, config grid, end-to-end judged evaluation with deterministic metrics + RAGAS + DeepEval (optional `eval` extra); v2 regression floor.
- Frontend rewritten: Next.js App Router workbench (login, home, investigations with evidence panel, documents, upload wizard with real stages, document detail, ingestion, settings, admin), shadcn/ui + Engineering Cobalt tokens, same-origin API, five Playwright flows.
- Infra: worker service in compose/entrypoint, models baked into the image, CI frontend + Playwright job with a seeded synthetic corpus.
- Removed: single-page v1 frontend components, `README_PACKAGE.md`; planning documents folded into README and ADR-0029.

## 0.2.0 — 2026-09-12 (rebuild)

Complete restructure from the CAE investigation workstation into a regulatory knowledge system. The audit, classification and measurements of that rebuild are in git history (`docs/rebuild-ledger.md` up to tag `pre-v2-product-rebuild`).

- New package `safety_assistant` (src layout): canonical regulatory model, lifecycle ingestion with idempotency keys and embedding reuse, regulation-aware structural parsing/chunking, hybrid retrieval (dense + BM25 + exact leg → RRF → rerank), temporal scoping, grounded generation under a citation contract with programmatic validation, bounded LangGraph agent (comparison, change analysis), RBAC/OIDC, SSRF-safe fetcher, LLM data-class policy, rate limiting, OpenTelemetry + Prometheus, hardened Docker, CI/security/eval/release workflows, Terraform, evidence-first Next.js UI with Playwright tests.
- Evaluation dataset `regulatory_v1` (47 cases, 16 slices) and per-leg runner; measured full-pipeline R@10 0.901 / MRR 0.636 on 21,910 chunks.
- Removed: `apps/`, `packages/`, legacy tests and scripts, CAE data, planning/prompt markdown. Recoverable at tag `pre-rebuild-baseline`.

## 0.1.0 — pre-rebuild
Passive Safety CAE Investigation Agent (see tag `pre-rebuild-baseline`).
