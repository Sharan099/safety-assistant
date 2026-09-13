# Changelog

## 0.2.0 — 2026-09-12 (rebuild)

Complete restructure from the CAE investigation workstation into a regulatory knowledge system. See `docs/rebuild/v1-ledger.md` for the audit, baseline, classification and measurements.

- New package `safety_assistant` (src layout): canonical regulatory model, lifecycle ingestion with idempotency keys and embedding reuse, regulation-aware structural parsing/chunking, hybrid retrieval (dense + BM25 + exact leg → RRF → rerank), temporal scoping, grounded generation under a citation contract with programmatic validation, bounded LangGraph agent (comparison, change analysis), RBAC/OIDC, SSRF-safe fetcher, LLM data-class policy, rate limiting, OpenTelemetry + Prometheus, hardened Docker, CI/security/eval/release workflows, Terraform, evidence-first Next.js UI with Playwright tests.
- Evaluation dataset `regulatory_v1` (47 cases, 16 slices) and per-leg runner; measured full-pipeline R@10 0.901 / MRR 0.636 on 21,910 chunks.
- Removed: `apps/`, `packages/`, legacy tests and scripts, CAE data, planning/prompt markdown. Recoverable at tag `pre-rebuild-baseline`.

## 0.1.0 — pre-rebuild
Passive Safety CAE Investigation Agent (see tag `pre-rebuild-baseline`).
