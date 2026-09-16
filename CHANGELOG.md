# Changelog

## 0.8.0 — 2026-09-16 (self-service sign-up and sign-in)

- Identity: `POST /api/v1/auth/signup` and `/auth/login` — a passive-safety engineer can create their
  own account (role `engineer` in the default organization) and sign back in, alongside the existing
  OIDC and dev-login paths. Passwords are hashed with `hashlib.scrypt` (stdlib, memory-hard, random
  16-byte salt, self-describing encoding for a future parameter bump) in `security/passwords.py` —
  never stored, logged, or returned in the clear.
- Security hardening: sign-in returns the same message and status for a wrong password and an
  unknown email (no account-existence oracle); a per-account lockout (`LOGIN_MAX_ATTEMPTS`,
  `LOGIN_LOCKOUT_MINUTES`) persists on the user row so it holds across replicas without a shared
  cache and clears on the next success; a per-IP rate limit bounds signup/login attempts
  (`auth_rate_limited`, independent of the per-principal query limiter); signup and login require
  the CSRF header despite having no session yet, closing login-CSRF; minimum password length and a
  same-as-email-local-part check at signup.
- Migration `0007` (additive/nullable/defaulted: `users.password_hash`, `failed_login_attempts`,
  `locked_until`; forward/rollback round-trip tested).
- CLI: `safety-assistant users add --password` and `users set-password` for operator-provisioned
  password accounts.
- Frontend: the login page gets a sign-in/create-account toggle (email, password, and a name field
  for signup) above the existing OIDC button and dev-login fallback.
- Tests: `tests/unit/test_passwords.py` (hashing correctness, malformed-hash fail-closed, rehash
  detection), `tests/security/test_password_auth.py` (signup, duplicate email, weak password,
  login success/failure parity, lockout, CSRF, feature flag off), Playwright flow 8 (sign up →
  sign out → wrong password refused → sign back in).

## 0.7.0 — 2026-09-15 (engineer walkthrough fixes, 461-question evaluation)

Findings and their status: `docs/QA_REPORT_2026-09-15.md`.

- Retrieval: a query scoped to one regulation is one document for the diversity cap (its amendment sheet no longer squeezes the answering clause out); each leg's top-1 always enters the rerank pool; "ask this document" searches the selected document *plus* the regulations the question names (authorization unchanged); definition questions with the regulation's own quoted definition in evidence pass the ambiguity gate; full defined-phrase matching in the definition leg; `pedestrian head` / `head impact` → `headform HIC`.
- Answers: prompt `grounded_v4` (stay on the asked regulation/criterion; state the clause complete with its conditions); numbers the engineer states in the question are accepted as known inputs by the validator (limits must still be in the evidence); evidence the provider is not cleared for is withheld by name while the cleared part is answered; an instruction with no regulatory content is declined before retrieval; 429 responses wait `Retry-After` (or 3 s / 6 s) inside the call budget.
- Ingestion: a missing OCR binary no longer fails text-layer uploads (scanned pages stay flagged); uploads are cited by title.
- Frontend: evidence panel only on investigations; verified-source badges name the kind; evidence-only turns explain themselves and offer *Ask again*; pipeline warnings in plain language; citation hover and panel highlight and scroll to the supporting lines; latest answer's evidence shown when an investigation opens; upload form labels and file sizes; document-focus scope label.
- Evaluation: gold `r94-003` / `r95-002` accept the twin clause of the sibling regulation; degenerate short-form variants dropped (131); the routed model is recorded per case; 461 questions measured (README "End-to-end answers").

## 0.6.0 — 2026-09-15 (engineer questions, small-model answers, production hardening)

- Answer path: the LLM call's timeout is a wall-clock budget including retries (≤ 30 s, then evidence-only); Next.js proxy timeout 120 s; a failed turn stays in the thread with Retry; citation markers and chips preview the cited lines, page and version on hover/focus.
- Answer quality: claim kind `CALCULATION` (derived values allowed only from evidence inputs, flagged "verify"); prompt `grounded_v3` (direct answer first, exact requirement sentence, scenario application, cross-document regimes, short forms, simulation questions, partial answers instead of blanket refusals); Settings → Current project (`user_preferences.project_context`) reaches the model as data; greetings and off-topic questions get a capabilities reply; `messages.abstain_reason` (migration `0006`).
- Retrieval: exact clause-identifier hits lead the result list and bypass the diversity cap (an amendment sheet counted as a second version and squeezed the clause out); a regulation's key selects its amendment sheets too; a deterministic definition leg for "what is X / X definition"; engineering synonyms for the lexical leg (webbing→strap …); ambiguity gate uses a passive-safety lexicon ("What is the maximum allowed value?" abstains).
- Evaluation: `engineer_scenarios_v1` (34 cases: scenarios, short forms, cross-document, market regimes, calculations, simulation, amendment awareness, declines); the judged harness fingerprints prompt text and pipeline code and never caches transient provider failures; gold cases that became answerable with the new corpus rewritten from corpus facts; `LLM_MODEL=gpt-oss-20b` measured as the small default (README).
- Deployment: production image rebuilt and smoke-tested (docs/OpenAPI/dev-login absent, unauthenticated 401, validator refuses insecure settings); API task memory 4 GB for two workers; Terraform fmt/validate; pip-audit and npm audit clean.

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
