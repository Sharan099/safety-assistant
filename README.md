# Safety Assistant — regulatory knowledge system for automotive passive safety

**Problem:** engineers need exact, current, citable answers from UN vehicle-safety regulations (R16, R94, R95, R129, …) — with the right *version*, the right *clause*, the numbers unchanged, and an honest "not in the corpus" when that is the truth. Generic PDF chatbots get the version wrong, invent clause numbers and round limits.

Safety Assistant is a versioned, auditable retrieval system: structure-aware ingestion of consolidated regulation texts, hybrid retrieval with temporal scoping, grounded generation under a citation contract, and programmatic validation of every claim.

## Demo

Run locally in about ten minutes (see [Local setup](#local-setup)). The UI shows, for every answer, the evidence id → regulation → version → section → page → validity window → source SHA-256, and states explicitly when it abstains.

```
$ curl -s localhost:8010/api/v1/ask -d '{"query":"What is the tibia index limit in UN R94?"}' -H 'content-type: application/json'
{ "mode": "EVIDENCE_ONLY", "citations": [ { "evidence_id": "E1", "label": "UN R94 Rev.4 §5.2.1.3–5.2.1.8 (pp. 12–13)",
  "version_label": "Rev.4 (04 series)", "valid_from": "2021-06-09", "version_status": "ACTIVE", "source_sha256": "acd8a96b…" } ], … }

$ curl -s localhost:8010/api/v1/ask -d '{"query":"What was the tibia index limit in UN R94 as of 2015?"}' …
{ "mode": "ABSTAINED", "abstain_reason": "no_version_valid_on_date",
  "answer": "No ingested version of UN-R94 was in force on 2015-12-31; …" }
```

## Measured results

All numbers below were produced by code in this repository on the stated corpus; nothing is estimated. Raw result files with git SHA, corpus fingerprint, model and config versions live in `evals/results/`.

**Retrieval** — dataset `regulatory_v1` (47 cases, 16 slices; 39 with section-level truth), corpus 21,910 chunks / 16 documents, git `f82ce5a`, 2026-09-12. Relevance = clause-path match; k_eval = 20.

| Pipeline leg | R@5 | R@10 | R@20 | P@5 | Hit@5 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| dense only (pgvector HNSW, all-MiniLM-L6-v2) | 0.597 | 0.793 | 0.831 | 0.159 | 0.718 | 0.523 | 0.555 | 130 |
| sparse only (BM25, stemmed) | 0.686 | 0.821 | 0.870 | 0.185 | 0.795 | 0.546 | 0.582 | 81 |
| hybrid RRF | 0.675 | 0.852 | 0.859 | 0.179 | 0.795 | 0.576 | 0.614 | 226 |
| hybrid RRF + reranker | 0.734 | 0.875 | 0.901 | 0.205 | 0.846 | 0.597 | 0.639 | 232 |
| **full** (+ exact-clause leg, parent/cross-ref expansion) | **0.759** | **0.901** | **0.926** | **0.210** | **0.872** | **0.636** | **0.671** | 269 |

Regulation-level hit@5 is 0.955 on every leg. The regression gate (`tests/retrieval_regression`) fails CI if the full-pipeline MRR drops below 0.60 or any of 23 stable cases loses its clause from the top 10.

**Load** — laptop (Intel i5-8250U, 8 GB), 2 uvicorn workers, no LLM, `scripts/eval/load_test.py`:

| Endpoint | Users | Duration | Requests | rps | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|---|---|
| `/api/v1/search` | 1 | 30 s | 97 | 3.2 | 286 ms | 398 ms | 429 ms | 0 |
| `/api/v1/search` | 5 | 60 s | 334 | 5.5 | 742 ms | 1.37 s | 7.9 s¹ | 0 |
| `/api/v1/ask` (evidence-only) | 5 | 45 s | 180 | 3.9 | 1.14 s | 2.39 s | 3.22 s | 0 |

¹ p99 includes the per-worker BM25 index build on first request (~3–5 s for 21,910 chunks); steady-state p99 is under 2 s.

**Generation / refusal** — the citation contract and abstention gates are verified by tests (mock LLM, schema output): hallucinated evidence ids and numbers absent from the cited text are rejected; no-version-on-date, unknown regulation and ambiguous queries abstain. No LLM-based answer-quality metrics are published because no LLM provider was available in this environment — see [Known limitations](#known-limitations).

**Ingestion** — 16 sources, 11,700 sections (275 typed definitions), 847 cross-references (648 resolved within the same text), 9,022 tables, 4,120 figures. Re-ingesting an unchanged source is a 1–2 s no-op; the synthetic v1→v2 update test re-embeds 2 of 12 chunks.

## Architecture

```
                  registry (allowlist, SHA-256, version dates)
                                  │
   ingest:  DISCOVERED → DOWNLOADED → VALIDATED → PARSED → NORMALIZED → CHUNKED → INDEXED → VERIFIED → ACTIVE
            fetch (SSRF-safe)  magic/size/hash  PyMuPDF   clause tree   structural   fastembed  consistency  atomic
            blob store (s3://) page limits      tables    annex scope   citation lbl HNSW       checks       supersede
                                  │
   PostgreSQL 16 + pgvector: regulations · regulation_versions · source_artifacts · sections · tables · figures
                             cross_references · chunks · chunk_embeddings · ingestion_runs/events · query_traces …
                                  │
   ask:     parse scope (regulation, clause, as-of date) → authz data classes → SQL scope (status + validity)
            → dense top-30 ∥ BM25 top-30 ∥ exact-clause leg → RRF → rerank → guard/diversify
            → parent + cross-ref expansion → evidence gate (abstain / one rewrite / proceed)
            → LLM under schema (evidence ids) → citation + numeric validation → QueryTrace
                                  │
   FastAPI (RBAC scopes, rate limit, request ids, OTel, Prometheus)  ←  Next.js evidence-first UI
```

The bounded agent (`agents/graph.py`, LangGraph) adds two routes on top of the standard path: **comparison** (one scoped retrieval per named regulation) and **change analysis** (section-level diff between the two latest versions as data for the model). Budgets — retrieval attempts, LLM calls, tool calls, wall-clock — are enforced in code.

## Why these decisions

| Decision | Why | ADR |
|---|---|---|
| PostgreSQL + pgvector as the only store | one system of record for metadata, audit and vectors; HNSW meets the measured latency; no second database to keep consistent | [0019](docs/ADR/0019-canonical-store-and-schema-cutover.md) |
| Structural chunks (clause tree), not fixed windows | citations must name exact clauses; merged tiny siblings keep numbers inline; tables carry headers | [0020](docs/ADR/0020-structural-chunking.md) |
| Dense + BM25 with reciprocal-rank fusion | measured: RRF beats either leg alone (MRR 0.576 vs 0.523/0.546); ranks fuse, raw scores are never summed | [0021](docs/ADR/0021-hybrid-retrieval-and-rrf.md) |
| Heuristic reranker by default, cross-encoder opt-in | +0.02 MRR / +0.025 nDCG for ~5 ms; the cross-encoder costs ~3.5 s/query on CPU | [0022](docs/ADR/0022-reranking.md) |
| Versions with validity windows + lifecycle states | "latest" means latest *in force*, never latest downloaded; historical queries filter before ranking | [0023](docs/ADR/0023-temporal-regulation-model.md) |
| Citation contract with programmatic validation | the model may not invent ids, pages or numbers; violations are dropped, not trusted | [0024](docs/ADR/0024-citation-contract.md) |
| Bounded LangGraph, no swarm | routing/decomposition is deterministic; the LLM only synthesises under schema | [0025](docs/ADR/0025-bounded-agent.md) |
| Typed provider interfaces, fakes only in `test` | production refuses hashing embeddings, mock LLM, anonymous auth at startup | [0026](docs/ADR/0026-provider-abstraction-and-fakes.md) |
| Explicit data-class policy per LLM provider | confidential evidence never reaches a provider that is not cleared; policy is config, not a model-name heuristic | [0027](docs/ADR/0027-security-policy.md) |
| ECS Fargate + RDS + S3, no Kubernetes | one stateless service and two managed stores do not justify a cluster | [0028](docs/ADR/0028-deployment-architecture.md) |

## Ingestion, versioning, freshness

- **Registry** `knowledge/00_registry/sources.yaml` is the allowlist: regulation identity, kind, jurisdiction, authority level, data class, official URI, SHA-256/size, and the consolidated text's own version label, series, revision, publication and entry-into-force dates (human-reviewed; the cover-page parser cross-checks them and logs disagreements).
- **Idempotency keys**: parse = `source_sha256 + parser_version + parser_config_hash`; chunk = `parsed_hash + chunker_version + chunker_config_hash`; embed = `chunk_sha256 + model_version + dimensions`; index = `chunk_id + index_schema_version`. Unchanged content is never re-embedded — chunk content is version-independent so an amendment reuses every untouched clause's vector.
- **Lifecycle** is a state machine (`domain/regulations/lifecycle.py`); only `ACTIVE` versions are retrievable for current queries, `ACTIVE|SUPERSEDED` for as-of queries. Activation supersedes the previous version in the same transaction and closes its validity window. Bad files are quarantined with an attempt budget; every transition is an `ingestion_events` row.
- **Freshness**: `sa_freshness_lag_days{regulation}` (publication → activation) and `ingestion_runs.stats`.

## Retrieval pipeline

Details in [docs/retrieval-design.md](docs/retrieval-design.md). Highlights: SQL scope (status, validity, regulation, data class) before any ranking; exact identifiers (`5.2.1.8`, `Annex 3`) route to a dedicated leg with weight 2 in RRF; BM25 is a process-cached index with in-memory scope filtering; the heuristic reranker rewards literal overlap, authority and normative clauses for requirement-seeking questions; the diversification cap is scope-aware; parent sections and resolved cross-references are attached to each evidence item within a token budget.

## Evaluation

`evals/datasets/regulatory_v1.yaml` — 47 cases across 16 slices (exact clause, paraphrase, identifier, definition, numeric threshold, units, table/annex, exception, multi-clause, cross-reference, comparison, historical, change analysis, ambiguous, unanswerable, adversarial). Growth target: 200–500. `uv run safety-assistant eval-retrieval` measures every leg independently and records provenance. See [docs/evaluation.md](docs/evaluation.md).

## Security and governance

RBAC scopes on every route (`regulation:read`, `chat:query`, `confidential:query`, `document:ingest`, `audit:read`, `system:admin`) via API keys or OIDC/JWKS; authorization narrows the retrieval universe by data class *before* ranking; SSRF-safe fetcher (https, allowlist, public-IP DNS check per hop, size cap, conditional GET); file validation (magic bytes, size, page count, hash against the registry); prompt-injection signals recorded and never executed; explicit LLM data-class policy; per-principal rate limit; audit actor on privileged ingestion; secrets from the environment/secret manager only. Threat model: [docs/security-threat-model.md](docs/security-threat-model.md).

## Reliability and observability

Liveness/readiness with dependency health (an LLM outage degrades to evidence-only, it never flips readiness); reranker failure degrades to fused order; structured JSON logs with request ids; OpenTelemetry spans for retrieval, agent, LLM and ingestion; Prometheus metrics (`/metrics`) for request/stage latency, answer modes, citation-validation failures, retrieval no-hit, LLM calls/tokens, ingestion runs, freshness lag; alert rules and a dashboard in `infra/monitoring/`. Runbook: [docs/operations-runbook.md](docs/operations-runbook.md).

## Local setup

```bash
uv sync --extra s3                          # Python 3.12+, uv
docker compose up -d postgres               # pgvector on localhost:5433
cp .env.example .env
uv run safety-assistant migrate             # creates schema in DATABASE_URL (create the database first if needed)
# put the registered PDFs under knowledge/ (see sources.yaml), then:
uv run python scripts/maintenance/verify_registry.py
uv run safety-assistant ingest              # ~25 min for all 16 sources on a laptop; regulations alone ~2 min
uv run uvicorn safety_assistant.api.main:app --port 8010
cd frontend && npm install && cp .env.local.example .env.local && npm run dev   # http://localhost:3010
```

Quality gates: `make lint types test` (119 tests; PostgreSQL required for integration/e2e), `make eval`, `make load`, `cd frontend && npm run test:e2e`.

## Deployment

- Image: `infra/docker/Dockerfile` (multi-stage uv build, non-root, read-only root fs, embedding model baked in, healthcheck). Full local stack with MinIO: `docker compose -f infra/docker/compose.yaml up --build`.
- CI: `.github/workflows/ci.yml` (ruff, mypy, all test tiers, container build, Trivy, SBOM), `security.yml` (pip-audit, bandit, gitleaks, semgrep), `eval.yml` (nightly retrieval evaluation; real corpus restored from a bucket when configured), `release.yml` (ghcr image with provenance + SBOM on tags).
- Infrastructure: `infra/terraform/` — ALB (TLS 1.3) → ECS Fargate service with circuit-breaker rollback → RDS PostgreSQL 16 (TLS forced, encrypted, 14-day backups/PITR, multi-AZ in production) + versioned, encrypted S3 artifacts + Secrets Manager + CloudWatch alarms. Container: `APP_ENV=production`, `AUTH_MODE=oidc`.

## Known limitations

- **No LLM-judged generation metrics.** No LLM provider was reachable while building this; generation was verified with a schema-compliant mock. Groundedness/correctness/refusal rates on the gold set are therefore not published. The harness (`AnswerService` + `evals/datasets`) is ready; run it with a configured provider before making quality claims.
- **One version per regulation in the corpus.** Historical and change-analysis behaviour is proven end-to-end on a synthetic two-version regulation (`tests/e2e`), not yet on two real UNECE consolidations; ingesting an earlier revision is a registry entry away.
- **Dataset size** is 47 cases (target 200–500); one case (`r129-003`) is still `DRAFT`.
- **Parser**: PyMuPDF only; scanned pages are flagged `NEEDS_REVIEW`, not OCR'd; table extraction is best-effort (drawings detected as tables are filtered). Docling/OCR plug in behind `DocumentParser`.
- **Official source URIs** in the registry are landing pages marked `LANDING_PAGE_UNVERIFIED`; the fetcher is only used for entries marked `VERIFIED`.
- **Rate limiter and BM25 index are per process** (documented ponytail ceilings); a shared limiter/index is the upgrade path beyond a few replicas.
- **Terraform is unapplied** in this repository (no cloud account); it is validated for structure, not by a real `apply`.
- Supporting documents (LS-DYNA manuals, NHTSA reports) are in the corpus as `MANUAL`/`TECHNICAL_REPORT`; they are never presented as regulations.
