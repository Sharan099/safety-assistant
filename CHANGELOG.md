# Changelog

## 0.5.0 — 2026-09-14 (curated 42-source corpus, Sources page)

- Corpus replaced by the delivered `Knowledge source` set, curated by `scripts/maintenance/build_registry.py`: 52 files → 42 sources (5 byte-identical duplicates, one older revision and one already-incorporated amendment sheet dropped; three scanned texts merged from their parts and OCR'd; amendment sheets newer than a consolidated text kept as separate sources). Files renamed by regulation, revision, series, year and subject under `knowledge/{unece,us_fmvss,euro_ncap,standards,cae_manuals,reference}`; registry fields (symbol, revision, series, dates) parsed from cover pages, null when absent. `scripts/maintenance/prune_corpus.py` removes documents that left the registry and, with `--superseded`, non-active versions.
- Frontend: **Sources** page (`/app/sources`) — the verified corpus grouped (UNECE in regulation-number order with amendment sheets under their text, FMVSS, Euro NCAP, CAE manuals, references) with version in force and status; Playwright flow 6.
- Evaluation: the two gold cases that targeted removed NHTSA reports (`doc-005`, `doc-006`) dropped from v1/v2 (45 / 260 cases); results re-measured on the new corpus (README "Measured results").

## 0.4.0 — 2026-09-13 (summary-augmented chunking)

Document identity in the retrieval representation, never in the evidence (ADR-0030).

- `document_summaries`: one generated retrieval summary per document version, cached by (artifact sha256, prompt version, model), validated (reasoning dumps, truncation and markdown are rejected and recorded as FAILED, retryable), provider data-class policy applied; `chunks.retrieval_text` = identity block + summary + unchanged content; `chunk_embeddings.representation` (`content` | `sac_v1`) with one partial HNSW index each. Migration `0005` (additive; downgrade restores the single index).
- `safety-assistant reindex` builds the sac_v1 index for an existing corpus: per-version commit, resumable, idempotent, failure reporting, coverage check. `SAC_ENABLED` builds it at ingest; `RETRIEVAL_REPRESENTATION` / `RETRIEVAL_SAC_SPARSE_WEIGHT` select what the query side uses. Chunk content, citations, the exact-clause leg and the temporal filter are untouched; `Evidence` has no field for the summary.
- Evaluation: document-level metrics (recall@1/3/5, MRR), the document-level retrieval mismatch rate (`drm@1`, `drm@5`, definition in `evaluation/retrieval_eval.py`), version hit, `document_level_retrieval_mismatch` in the generation failure taxonomy, `source: synthetic` provenance, `hard_negative_regulation_keys`; benchmark `evals/datasets/document_mismatch_v1.yaml` (36 twin-clause cases); `scripts/eval/sac_ab.py` (A/B on identical queries and labels) and `scripts/eval/drm_cases.py` (per-case flips); retrieval traces carry document/version/section and a document distribution.
- Measured results and the adopted configuration: README "Measured results".

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
