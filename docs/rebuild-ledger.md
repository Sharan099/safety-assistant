# Rebuild Ledger — Automotive Safety RAG / Safety Assistant

Single authoritative progress document for the production rebuild governed by
`CLAUDE.md`. Updated at every milestone. No separate per-phase status files.

## 0. Recovery checkpoint

| Item | Value |
|---|---|
| Pre-rebuild commit | `af051f2c7b4b9de8852d84133140ba833e18c0cc` (`feat: three-panel investigation workspace shell`) |
| Recovery tag | `pre-rebuild-baseline` (points at the commit above) |
| Rebuild branch | `main` (repository workflow is direct commits on `main`; no remote, no prior tags/branches) |
| Rebuild start | 2026-09-11 |
| Contract | `CLAUDE.md` (rewritten by the user 2026-09-11, committed with this ledger) |
| Uncommitted user work at start | `CLAUDE.md` (modified), `CLAUDE_KICKOFF_PROMPT.md`, `PRODUCTION_REBUILD_CHECKLIST.md` (untracked) — all captured in the M0 commit |

Rules honoured: no force-push, no `.git` deletion, no history rewrite.

Environment notes: `.claude/skills/` does **not** exist in this repository (CLAUDE.md §19
references `repo-audit`, `rag-architecture`, … — none are present). Work proceeds without them.

## 1. Baseline architecture (as found)

Product as found: **"Passive Safety CAE Investigation Agent"** — a CAE investigation
workstation (compare two crash-simulation runs → quality gate → comparability →
config diff → signal analysis → evidence → hypothesis → engineer review) with a
**knowledge layer** (PDF ingestion + BM25/pgvector hybrid retrieval) and an
investigation **Copilot** (LangGraph, SSE streaming). Regulatory retrieval is one
tool inside this workstation, not the product.

```
apps/api            FastAPI (runs, investigations, knowledge/search, copilot SSE)  823 LOC
apps/web            Next.js 16 investigation workspace (port 3010)               1,745 LOC TS
packages/domain     SQLAlchemy models (55 tables) + 4 Alembic migrations        2,206 LOC
packages/ingestion  PyMuPDF PDF pipeline, QA report, tables/figures, archives   1,448 LOC
packages/retrieval  BM25 + pgvector + RRF + reranker + relevance guard            744 LOC
packages/agent      LLMProvider, investigation LangGraph, Copilot LangGraph     2,015 LOC
packages/analysis   deterministic CAE analysis (quality, comparability, signals) 1,242 LOC
packages/cae        LS-DYNA lexer/parser/include graph/keyword scan               979 LOC
scripts/            7 top-level scripts (synthetic data, ingest, index, OKF)      813 LOC
evals/              5 harnesses + 8-query golden set + 2 benchmark results        611 LOC
tests/              65 files, 210 tests                                          3,526 LOC
docs/ADR/           18 ADRs
root *.md           17 planning/prompt/status documents
```

Data stores: PostgreSQL 16 + pgvector (docker-compose, host port 5433). No Redis,
no queue, no object storage, no auth, no CI workflows, no Dockerfile for the app,
no IaC, no observability beyond stdout.

Retrieval as found: `retrieve()` = BM25 (in-memory rank-bm25 rebuilt per call over
*all* chunks) + pgvector cosine (fastembed all-MiniLM-L6-v2, 384-d) → RRF(k=60) →
`LexicalAuthorityReranker` (heuristic; real cross-encoder implemented, opt-in) →
lexical relevance floor + authority allowlist + per-document dedup → top-k.
`section_content` is returned as the parent-context unit.

## 2. Baseline measurements (2026-09-11, HEAD af051f2)

| Check | Result |
|---|---|
| `uv run ruff check .` | pass |
| `uv run mypy apps packages scripts tests conftest.py evals` (strict) | pass, 143 files |
| `uv run pytest` | **209 passed, 1 skipped**, 141.7 s (needs Postgres; self-seeds `passive_safety_test`) |
| `evals/retrieval_eval.py` (8 queries, doc-level) | Recall@5 1.00, Recall@10 1.00, MRR 0.900 |
| `evals/level3_hybrid_eval.py` BM25 only | R@5 0.88, R@10 1.00, MRR 0.706 |
| … Dense only | R@5 0.75, R@10 0.88, MRR 0.671 |
| … BM25+Dense RRF | R@5 0.88, R@10 0.88, MRR 0.667, nDCG@10 0.720 |
| … + heuristic reranker (production path) | R@5 1.00, R@10 1.00, MRR 0.900, nDCG@10 0.923 |
| Docker build | not applicable — no application Dockerfile exists |
| CI | none configured |

Dev corpus (`passive_safety` DB): 20 knowledge_sources, 16 documents, 34 revisions,
10,309 pages, 9,641 sections, 11,921 chunks, 12,871 embeddings, 9,033 tables,
4,235 figures; 4 CAE decks, 10,210 keywords, 1,996 parts.

**Discrepancies found during baseline**

- README publishes MRR 0.917 (BM25) / 0.838 (dense) / 0.806 (RRF) / 0.938 (full) and nDCG 0.954.
  Those were measured on a ~475-chunk corpus; on the current 11,921-chunk corpus they are
  0.706 / 0.671 / 0.667 / 0.900 / 0.923. README numbers are stale → must be regenerated, never copied.
- Same document has several `READY` revisions (`-first20`, `-first60`, `-full`). `retrieve()` does
  not filter by revision/active state, so truncated partial revisions compete with full ones in
  the ranked list. There is no `ACTIVE` concept. (Violates CLAUDE.md §6 "only ACTIVE content".)
- Golden set is 8 document-level cases; no section/chunk ground truth, no slices, no
  unanswerable/adversarial cases, no as-of-date.
- BM25 index is rebuilt from every chunk on every query (O(corpus) per request) — fine at 475
  chunks, a latency problem at 12k. Must be measured, then fixed (persist or cache).
- `Embedding.embedding` is `Vector()` with no fixed dimension and no ANN index.
- Regulation identity, version, jurisdiction, effective date, supersession: **not modelled**
  (`DocumentRevision.effective_date` exists but is never populated; `revision_label` is the
  extractor version, not a regulatory version).
- Provenance is partial: `SourceSnapshot.sha256` + manifest hash check exist (good);
  parser/chunker/embedding config hashes are not recorded; no source URI/ETag/Last-Modified.
- `knowledge/00_registry/source_manifest.yaml` still says `processing_status: NOT_INGESTED` for
  everything; `scripts/verify_manifest.py` referenced by the manifest does not exist.
- `packages/ingestion/pipeline.py` hard-codes `REPO_ROOT/data/artifacts` — container filesystem
  as artifact store.
- No auth of any kind; `apps/api/deps.py` attributes everything to a seeded default user.
- LLM path: `FreeLLMAPIProvider` (local proxy), no key configured in this environment; every
  agent test runs with `MockProvider`/no LLM. Generation is not evaluated anywhere.

## 3. Inventory and classification

Legend: `KEEP` (move as-is), `MIGRATE` (move + adapt to new package/model),
`REWRITE` (replace, preserve behaviour via characterization tests), `ARCHIVE`
(keep only for audit value, out of active tree), `DELETE`.

### 3.1 Backend code

| Path | Classification | Notes |
|---|---|---|
| `packages/retrieval/search.py` | MIGRATE → `retrieval/{hybrid,fusion,filters,service}.py` | RRF, dedup, guard proven; add revision/temporal/authz filters, candidate sizes from config |
| `packages/retrieval/bm25.py` | MIGRATE → `retrieval/sparse.py` | keep tokenizer (`*MAT_024`); add cached/persisted index |
| `packages/retrieval/embeddings.py` | MIGRATE → `providers/embeddings/` | keep FastEmbed + Hashing(test-only); readiness fails if load fails |
| `packages/retrieval/rerank.py` | MIGRATE → `providers/rerankers/` + `retrieval/rerank.py` | both rerankers proven and benchmarked |
| `packages/retrieval/relevance.py` | MIGRATE → `retrieval/filters.py` | pure functions, keep |
| `packages/retrieval/index.py` | MIGRATE → `ingestion/index/` | idempotent per (chunk, model, version) — keep semantics, key on chunk_sha256 |
| `packages/retrieval/structured.py` | see §3.7 scope decision | CAE structured search |
| `packages/ingestion/extract.py` | MIGRATE → `ingestion/parse/pymupdf.py` behind parser Protocol | NUL-strip + quality heuristics proven at 10k pages |
| `packages/ingestion/qa.py` | KEEP → `ingestion/validation/` | no-silent-loss report, fault-injection tested |
| `packages/ingestion/structure.py` | MIGRATE → `ingestion/parse/` | tables/figures; table rows must carry headers/caption (currently raw rows JSON) |
| `packages/ingestion/sections.py` | REWRITE → `ingestion/normalize/` | regex heading heuristic; must become regulation-aware (clauses `5.2.1.`, Annexes, definitions) with golden tests |
| `packages/ingestion/chunking.py` | REWRITE → `ingestion/chunk/` | word-count packing; needs structural parent/child, token-aware, stable chunk_sha256 |
| `packages/ingestion/pipeline.py` | REWRITE → `ingestion/workflows/` | lifecycle states, fingerprints, idempotency keys, quarantine, atomic activation, object-storage abstraction |
| `packages/ingestion/manifest.py` | MIGRATE → `ingestion/sources/registry.py` | manifest = source allowlist; add URI/authority/regulation identity |
| `packages/ingestion/archives.py` | KEEP → `ingestion/validation/archives.py` | zip-slip/bomb guards, tested |
| `packages/ingestion/profiling.py`, `router.py` | MIGRATE | corpus profiler + PDF router |
| `packages/domain/knowledge.py` | REWRITE → `persistence/models/` | becomes regulations / regulation_versions / source_artifacts / sections / tables / figures / cross_references / chunks; data migration from documents/document_revisions |
| `packages/domain/provenance.py` | MIGRATE | processing_jobs/steps → ingestion_runs/ingestion_events |
| `packages/domain/base.py`, `db.py` | MIGRATE → `persistence/` + `config/` | split Settings from engine |
| `packages/domain/migrations/*` | KEEP (history) + new migrations | never rewrite applied migrations |
| `packages/agent/llm.py` | MIGRATE → `providers/llm/` | keep no-silent-fallback rule; add schema output, error classification (429 vs 5xx) |
| `packages/agent/copilot*.py`, `graph.py`, `tools.py`, `evidence.py` | see §3.7 | CAE investigation agent |
| `packages/analysis/*` | see §3.7 | deterministic CAE analysis |
| `packages/cae/*` | see §3.7 | LS-DYNA parser |
| `apps/api/main.py`, `deps.py`, `schemas.py` | REWRITE → `api/` | add auth deps, middleware, health semantics |
| `apps/api/routers/knowledge.py` | MIGRATE → `api/routes/` | |
| `apps/api/routers/{runs,investigations,copilot}.py`, `apps/api/parquet_io.py` | see §3.7 | |
| `conftest.py` (root) | MIGRATE → `tests/conftest.py` | test-DB bootstrap is valuable; seeding must use new pipeline |

### 3.2 Scripts

| Path | Classification |
|---|---|
| `scripts/ingest_documents.py`, `ingest_level3_pdfs.py`, `index_knowledge.py` | REWRITE → `scripts/maintenance/ingest.py` (one CLI over new workflow) |
| `scripts/profile_knowledge_sources.py` | MIGRATE → `scripts/maintenance/` |
| `scripts/generate_okf_concepts.py` | ARCHIVE (OKF markdown is a doc artefact, not runtime) |
| `scripts/generate_synthetic_dataset.py`, `ingest_level3_cae_decks.py` | see §3.7 |

### 3.3 Tests (210) — all KEEP or MIGRATE; none deleted or weakened

| Dir | Files | Classification |
|---|---|---|
| `tests/retrieval/` | 6 | MIGRATE → `tests/unit` + `tests/retrieval_regression` |
| `tests/ingestion/` | 8 | MIGRATE → `tests/ingestion` + `tests/parser_golden` |
| `tests/domain/` | 3 | MIGRATE → `tests/integration` |
| `tests/api/` | 6 | MIGRATE → `tests/integration` / `tests/e2e` |
| `tests/agent/`, `tests/analysis/`, `tests/cae/` | 19 | see §3.7 |
| `tests/test_level3_smoke.py`, `test_source_manifest.py`, `test_environment.py` | 3 | MIGRATE |

### 3.4 Evaluation artefacts

| Path | Classification |
|---|---|
| `evals/golden_retrieval_set.yaml` (8 cases) | MIGRATE → `evals/datasets/` with full case schema; grow toward 200–500 |
| `evals/retrieval_eval.py`, `level3_hybrid_eval.py` | REWRITE → `safety_assistant/evaluation/` + `scripts/eval/` (R@k, P@k, HitRate, MRR, nDCG; record git SHA/config) |
| `evals/embedding_benchmark.py`, `reranker_benchmark.py` + `results/*.json` | KEEP → `evals/experiments/` (E5, E3); results → `evals/baselines/` |
| `evals/scenario_eval.py` | see §3.7 |

### 3.5 Docs / Markdown (root)

| File | Classification | Reason |
|---|---|---|
| `CLAUDE.md`, `README.md` | KEEP / REWRITE | README must reflect final architecture + measured results only |
| `PRD.md`, `TRD.md`, `APP_FLOW.md`, `BACKEND_SCHEMA.md`, `IMPLEMENTATION_PLAN.md`, `UI_UX_DESIGN_BRIEF.md`, `ENVIRONMENT_SETUP.md` | ARCHIVE → `docs/archive/` | V1 product docs; superseded by docs/architecture.md etc. |
| `PRD_LEVEL3(1).md`, `TRD_LEVEL3.md`, `CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md`, `PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md`, `UI_UX_DESIGN_BRIEF_LEVEL3.md`, `PRD_COPILOT_UPDATE.md`, `CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md`, `CLAUDE_CODE_BOOTSTRAP_PROMPT.md` | DELETE at cutover (recoverable via tag) | agent-session instruction prompts / superseded plans |
| `CLAUDE_KICKOFF_PROMPT.md`, `PRODUCTION_REBUILD_CHECKLIST.md` | checklist folded into §6; both DELETE at cutover | |
| `docs/ADR/0001–0018` | KEEP (new ADRs continue at 0019) | real decisions with evidence |
| `apps/api/README.md`, `packages/*/README.md`, `tests/README.md`, `scripts/README.md`, `evals/README.md`, `data/README.md`, `knowledge/0x/README.md` | DELETE / fold into docs | per-package narration |
| `apps/web/AGENTS.md`, `apps/web/CLAUDE.md`, `apps/web/README.md` | DELETE | Next.js scaffold boilerplate |

### 3.6 Data, generated files, secrets, machine-specific paths

| Item | Classification | Notes |
|---|---|---|
| `knowledge/**/*.pdf`, `Knowledge source/` | KEEP (gitignored) | immutable raw corpus; SHA-256 in manifest |
| `data/artifacts/**` (110 MB, gitignored) | regenerate under object-storage abstraction | extraction reports have audit value → keep regenerable |
| `data/parquet/**` (tracked) — synthetic CAE signals | see §3.7 | |
| `knowledge/07_okf/**/index.md` | ARCHIVE | generated concept files |
| `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `__pycache__/` | DELETE (already ignored) | |
| Secrets | none found; `change_me` Postgres password in compose/Settings defaults → dev-only, must not be a production default | |
| Machine-specific | port 5433/3010/8010 rationale "sibling project on this host" (ADR-0004, compose comment, Settings) → make configurable, drop host narration; `C:\…` paths: none in tracked code | |

### 3.7 Scope decision required: CAE investigation subsystem

`CLAUDE.md` defines the product as a regulatory knowledge system and its target
tree has no place for `analysis/`, `cae/`, investigations, runs, signals, synthetic
scenarios, or the investigation Copilot (~6,300 LOC backend + the whole frontend +
~45 of 55 tables + ~110 tests). These are working, tested and the most developed
part of the repository. Options:

- **A — Regulatory-only product (what the contract describes).** CAE subsystem is
  frozen at tag `pre-rebuild-baseline`, removed from the active tree at cutover,
  documented in this ledger as archived. One clear product. Least ambiguity, least
  ongoing maintenance, matches CLAUDE.md §4 and the kickoff's "one obvious
  production package".
- **B — Keep CAE as a bounded secondary module** inside `src/safety_assistant/`
  (`cae/`, `investigation/`) with its own routes/tests, sharing providers/persistence.
  Preserves everything; roughly doubles migration effort; README/architecture must
  explain two products.

**Decision (user, 2026-09-11): A — regulatory-only product.** CAE subsystem
(`packages/analysis`, `packages/cae`, investigation/copilot agent, runs/investigations
API, synthetic scenarios, `apps/web` investigation UI, `data/parquet`, CAE tests) is
frozen at tag `pre-rebuild-baseline` and will be removed at cutover (M16).

## 4. Milestone log

| Milestone | Status | Evidence |
|---|---|---|
| M0 audit + baseline + checkpoint | **done 2026-09-11** | this ledger §1–3; tag `pre-rebuild-baseline`; tests 209/1 skipped; evals above |
| M1 packaging / boundaries | **done** | `src/safety_assistant/` (src layout, hatchling), `config/settings.py` profiles (production refuses hashing/mock/no-auth/dev password), typed providers, `migrations/` chain 0001 |
| M2 canonical regulatory model | **done** | `persistence/models/regulatory.py` + `operations.py` (14 tables), `domain/regulations/lifecycle.py` state machine; migration round-trips up/down/up |
| M3 incremental ingestion | **done** | `ingestion/workflows/ingest.py`: DISCOVERED→…→ACTIVE, parse/chunk/embed idempotency keys, `SKIPPED_UNCHANGED` no-op re-runs (measured 1.3 s), quarantine + attempt budget, atomic supersession; blob store content-addressed; 16/16 sources ACTIVE, 18,599 chunks |
| M4 structural parsing/chunking | **done** | `normalize/structure.py` clause tree with annex scope, TOC/footnote guards, definitions (R94 63, R16 71, R95 52, R129 89), cross-refs (R94 127 → 100 resolved); `chunk/structural.py` exact citation labels, merged tiny siblings keep inline numbers, table chunks carry headers |
| M5 hybrid retrieval | **done** | `retrieval/service.py`: SQL scope before ranking, dense (HNSW) + BM25 (cached, in-memory scope) + exact-identifier leg → RRF → normative-aware heuristic rerank → guard/diversify → parent + cross-ref expansion; ~200–250 ms/query p50 |
| M6 evaluation baseline | **done** | `evals/datasets/regulatory_v1.yaml` 47 cases / 16 slices (39 section-level); `scripts/eval/retrieval.py`; results in `evals/results/`; see §7 |
| M7 grounded generation | **done** | `generation/`: GroundedDraft schema, gate (ambiguous / no-version-on-date / unknown regulation / no evidence / weak + one acronym rewrite), citation + numeric validation, modes GENERATED/EVIDENCE_ONLY/ABSTAINED, QueryTrace per request; API `/ask`, `/search`, `/evidence/{id}`, `/regulations`, `/feedback`, admin ingest/audit; RBAC scopes; health live/ready/deps |
| M8 temporal RAG | **done** | validity windows + lifecycle in SQL scope; `as_of` parsing; `/regulations/{key}/versions?as_of`; e2e: current → v2, as-of → v1, before v1 → abstain |
| M9 change-impact | **done** | `ingestion/diff/sections.py` (path + content-hash diff, unified text), `/regulations/{key}/diff`, agent change_analysis route; e2e on synthetic v1→v2 |
| M10 bounded agent | **done** | `agents/graph.py` LangGraph: parse → route (standard / comparison / change_analysis) → gate → one rewrite → generate → validate; Budget (3 retrievals, 1 LLM call, 8 tools, 45 s); typed tools |
| M11 security/RBAC | **done** | api_key + OIDC/JWKS, 5 roles → scopes, authz narrows data classes before ranking; SSRF-safe fetcher (ETag/If-Modified-Since); LLM data-class policy; rate limit; audit actor; 30 security tests |
| M12 observability | **done** | JSON logs with request ids, OpenTelemetry spans (retrieval/agent/llm/ingest), Prometheus `/metrics` (request/stage latency, answer modes, citation failures, no-hit, LLM calls/tokens, ingestion, freshness lag), alerts + dashboard in `infra/monitoring` |
| M13 perf/load/fault | **done** | fault injection tests (reranker crash → degraded, embedding crash → 503/500 with request id, DB down → 503, event loop non-blocking); load: /search 5 users 5.5 rps p50 742 ms p95 1.37 s, 0 errors (see §8) |
| M14 deployment/IaC | **done (Terraform unapplied)** | multi-stage non-root image, compose stack with MinIO, CLI, S3 blob store, CI/security/eval/release workflows, Terraform ALB+ECS Fargate+RDS+S3+alarms |
| M15 frontend | **done** | `frontend/` Next.js evidence-first UI; Playwright 3/3 against the live API |
| M16 docs + cleanup | **done** | cutover commit f0281de removed apps/, packages/, tests/legacy, old scripts/evals, 17 root markdown files; README, SECURITY, CONTRIBUTING, CHANGELOG, LICENSE, docs/*.md, ADR-0019…0028 |
| M17 full validation | **done** | see §9 final gates |

## 6b. Test pyramid (new suite, `uv run pytest`; legacy suite `uv run pytest tests/legacy -o addopts=""`)

| Category | Location | Tests | Needs |
|---|---|---|---|
| unit | `tests/unit` | 53 | nothing |
| parser golden (real UN R94/R16 page text fixtures) | `tests/parser_golden` | 9 | nothing |
| security / adversarial | `tests/security` | 24 | PostgreSQL for 5 |
| integration (API `/ask` with mock LLM over synthetic corpus) | `tests/integration` | 7 | PostgreSQL |
| e2e (lifecycle, idempotent rerun, v1→v2 update with 2/12 re-embeds, quarantine) | `tests/e2e` | 4 | PostgreSQL |
| evaluation (dataset + report provenance) | `tests/evaluation` | 3 | nothing |
| retrieval regression (real corpus; skipped otherwise) | `tests/retrieval_regression` | 3 | ingested corpus |
| **total new** | | **98 passed** (2026-09-11) | |
| legacy (pre-rebuild, deleted at cutover) | `tests/legacy` | 209 passed / 1 skipped | legacy DB |

## 7. Measured retrieval results (new system)

Dataset `regulatory_v1`, 47 cases (39 with section-level truth), corpus 18,599 chunks,
git e537783+, 2026-09-11. Relevance = section path match (chunk level), k_eval=20.

| leg | R@5 | R@10 | R@20 | P@5 | Hit@5 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| dense only | 0.655 | 0.775 | 0.836 | 0.179 | 0.769 | 0.530 | 0.558 | 113 |
| sparse only (BM25) | 0.673 | 0.834 | 0.870 | 0.179 | 0.769 | 0.560 | 0.597 | 66 |
| hybrid RRF | 0.700 | 0.877 | 0.901 | 0.179 | 0.821 | 0.598 | 0.637 | 195 |
| hybrid RRF + rerank | 0.753 | 0.875 | 0.901 | 0.200 | 0.872 | 0.625 | 0.660 | 205 |
| full (+ exact leg, parent/xref) | **0.779** | **0.901** | **0.926** | 0.205 | **0.897** | **0.664** | **0.692** | 254 |

Iteration record (full leg, same dataset): first run MRR 0.627 / R@10 0.761 →
light stemming + short-query guard + normative-aware rerank 0.657 / 0.825 →
scope-aware diversification cap 0.664 / 0.901. Weakest slices: numeric_threshold MRR 0.505
(annex computation text competes with the body clause), multi_clause_reasoning (n=1).
Note the earlier README's "RRF dilutes" observation does not hold here: RRF beats both legs.

## 5. Decisions and deletions log

- 2026-09-11 — Tag `pre-rebuild-baseline` created at af051f2. Rebuild continues on `main`.
- 2026-09-11 — Existing ADR numbering retained; new ADRs start at 0019.
- 2026-09-11 — New schema lives in a new database (`safety_assistant`), new Alembic chain under `migrations/`; the legacy `passive_safety` DB and `packages/domain/migrations` are left untouched for recovery (ADR-0019).
- 2026-09-11 — Embedding column fixed at 384-d (all-MiniLM-L6-v2) with HNSW index; a different model is a schema migration by design (ADR-0019).
- 2026-09-11 — LS-DYNA manuals / NHTSA reports stay in the corpus as `kind=MANUAL|TECHNICAL_REPORT|STANDARD` supporting documents (never `REGULATION`); registry `sources.yaml` schema v2 replaces `source_manifest.yaml` at cutover.
- 2026-09-11 — `packages/ingestion/archives.py` + `profiling.py` (archive/LS-DYNA corpus profiler) classified ARCHIVE under decision A (no archive ingestion in the regulatory product).

## 6. Definition-of-Done checklist (from PRODUCTION_REBUILD_CHECKLIST.md, tracked here)

Unchecked until evidence is committed. See CLAUDE.md §22 for the authoritative list.

- [ ] One active application architecture / no duplicate backends / no stale entrypoints
- [ ] Immutable source artifacts, regulation/version separation, dates, supersession
- [ ] Section/table/cross-ref structure, fingerprints, incremental re-index, idempotency, quarantine, atomic activation, freshness
- [ ] Dense + sparse + RRF + rerank + metadata/temporal filters + parent expansion, regression in CI
- [ ] Structured output, evidence IDs, citation validation, conflict handling, abstention, no fake confidence
- [ ] Versioned dataset, slices, R@k/MRR/nDCG, citation/groundedness/refusal metrics tied to SHA
- [ ] Auth/RBAC, privileged endpoints, authz-before-retrieval, upload/fetch validation, SSRF, parser limits, injection suite, provider policy, cache policy, scans
- [ ] Liveness/readiness, timeouts/retries/backoff, DLQ, structured logs, OTel, metrics, SLOs, load/fault tests, backup/rollback
- [ ] Reproducible hardened Docker, IaC, release workflow, CHANGELOG


## 8. Final measurements (2026-09-12, git f82ce5a and later)

Corpus (`safety_assistant` DB): 16 regulations/documents, 16 versions all ACTIVE, 11,700 sections
(275 typed definitions), 21,910 chunks = 21,910 embeddings (384-d, HNSW), 9,022 tables, 4,120 figures,
847 cross-references (648 resolved). Regulations: UN R16 Rev.7 (393 chunks), R94 Rev.4 (277),
R95 Rev.3 (345), R129 Rev.3 (598).

Retrieval, `regulatory_v1` (47 cases / 39 with section truth), k_eval 20:

| leg | R@5 | R@10 | R@20 | P@5 | Hit@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.597 | 0.793 | 0.831 | 0.159 | 0.718 | 0.523 | 0.555 | 130 | 357 |
| sparse | 0.686 | 0.821 | 0.870 | 0.185 | 0.795 | 0.546 | 0.582 | 81 | 137 |
| hybrid RRF | 0.675 | 0.852 | 0.859 | 0.179 | 0.795 | 0.576 | 0.614 | 226 | 345 |
| + reranker | 0.734 | 0.875 | 0.901 | 0.205 | 0.846 | 0.597 | 0.639 | 232 | 502 |
| full | 0.759 | 0.901 | 0.926 | 0.210 | 0.872 | 0.636 | 0.671 | 269 | 366 |

Load (laptop i5-8250U, 2 uvicorn workers, no LLM, rate limit off):
/search 1 user 30 s: 97 req, 3.2 rps, p50 286 / p95 398 / p99 429 ms, 0 errors.
/search 5 users 60 s: 334 req, 5.5 rps, p50 742 / p95 1,373 / p99 7,918 ms (first-request BM25 build), 0 errors.
/ask evidence-only 5 users 45 s: 180 req, 3.9 rps, p50 1,137 / p95 2,389 / p99 3,223 ms, 0 errors.

Generation/refusal: verified by contract tests with a schema-compliant mock; **no LLM-judged metrics**
(no provider available). Frontend E2E: Playwright 3/3.

## 9. Final gates (2026-09-12)

| Gate | Result |
|---|---|
| `ruff check` / `ruff format --check` (src, tests, scripts, migrations) | pass |
| `mypy --strict` (src, scripts/eval, scripts/maintenance) | pass, 100 files |
| `pytest` (new suite) | **119 passed** — unit 53, parser golden 9, security 30, integration 16, e2e 5, evaluation 3, retrieval regression 3 |
| Retrieval regression gate | pass (MRR 0.636 ≥ 0.60 floor; 23/23 stable cases) |
| Playwright E2E | 3/3 |
| Frontend `tsc --noEmit`, `eslint` | pass |
| Container build + smoke | **executed 2026-09-12**: `infra/docker/Dockerfile` builds (281 MB, uid 10001); `APP_ENV=production` refuses `AUTH_MODE=none`; unauthenticated `/search` → 401; readiness green against the host database with a non-default role; `/ask` answers from the real corpus. Trivy/SBOM run in `ci.yml` only (not executed locally) |
| Dependency / secret / SAST scans | defined in `security.yml`; not executed here |
| Terraform | written, **not applied** |

## 10. Deletions and archives (cutover commit f0281de)

Deleted (recoverable at `pre-rebuild-baseline`): `apps/api`, `apps/web`, `packages/{domain,ingestion,retrieval,agent,analysis,cae}`,
`tests/legacy` (209 tests), `scripts/{generate_synthetic_dataset,ingest_documents,ingest_level3_pdfs,ingest_level3_cae_decks,index_knowledge,profile_knowledge_sources,generate_okf_concepts}.py`,
`evals/{retrieval_eval,level3_hybrid_eval,scenario_eval,embedding_benchmark,reranker_benchmark}.py`, `evals/golden_retrieval_set.yaml`,
`knowledge/00_registry/source_manifest.yaml`, `knowledge/07_okf/**`, `knowledge/0{3,4,5,6}_*/README.md`, `data/parquet`, `data/synthetic`,
root: `APP_FLOW, BACKEND_SCHEMA, CLAUDE_CODE_BOOTSTRAP_PROMPT, CLAUDE_CODE_COPILOT_CHANGE_REQUEST, CLAUDE_CODE_LEVEL3_INSTRUCTIONS,
CLAUDE_KICKOFF_PROMPT, ENVIRONMENT_SETUP, IMPLEMENTATION_PLAN, PASSIVE_SAFETY_LEVEL3_FINAL_FIX, PRD, PRD_COPILOT_UPDATE, PRD_LEVEL3(1), PRODUCTION_REBUILD_CHECKLIST, TRD, TRD_LEVEL3, UI_UX_DESIGN_BRIEF, UI_UX_DESIGN_BRIEF_LEVEL3` (.md), root `alembic.ini`.
Archived: `evals/baselines/pre-rebuild_{embedding,reranker}_benchmark.json`; ADR-0001…0018 kept as history.
Untouched on purpose: the pre-rebuild `passive_safety` database and the local `Knowledge source/` folder.

## 11. Known limitations (honest list)

See README "Known limitations": no LLM-judged generation metrics; one real version per regulation
(temporal behaviour proven on a synthetic two-version regulation); 47-case dataset (target 200–500),
one DRAFT case; PyMuPDF-only parsing (no OCR); registry source URIs unverified landing pages;
per-process rate limiter and BM25 index; Terraform unapplied; Trivy/SBOM/pip-audit/gitleaks/semgrep defined in CI but not executed locally.
