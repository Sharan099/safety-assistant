# Safety Assistant — passive-safety regulatory intelligence workspace

**Problem:** passive-safety engineers need exact, current, citable answers from UN vehicle-safety regulations (R16, R94, R95, R129, …) and from their own project documents — with the right *version*, the right *clause*, the numbers unchanged, and an honest "not in the sources" when that is the truth. Generic PDF chatbots get the version wrong, invent clause numbers, round limits and mix private material into shared answers.

Safety Assistant is a multi-user, evidence-first workbench: structure-aware asynchronous ingestion of regulations and uploads, document-level authorization evaluated before ranking, hybrid retrieval with a cross-encoder reranker and temporal scoping, grounded generation under a citation contract, programmatic validation of every claim, persistent investigations, and a UI where the evidence panel is the primary object.

## What it does

- **Sign in** (OIDC in production, seeded dev login locally); users belong to an organization and workspaces.
- **Ask** across *verified regulations*, *workspace documents*, *my private documents* or all authorized sources — the selection is validated against membership and enforced in SQL before any ranking.
- **Every answer** carries a mode badge (Grounded / Evidence only / Insufficient evidence), inline citation markers, and an evidence panel with regulation, version, clause, page, validity window, scope and the exact excerpt; abstentions explain what was searched.
- **Upload a PDF** → validation → parsing → chunking → embedding → indexing → verification → READY, run by a queue worker; the UI shows the real stage, public failure reasons with a diagnostic reference, and retry/replace. Uploads are private by default and never become authoritative without an audited promotion.
- **Resume** any investigation after logout: messages, citations and source scope persist; history is context for wording, never evidence.

## Measured results

All numbers were produced by code in this repository on the stated corpus; raw result files with git SHA, corpus fingerprint and config live in `evals/results/`. Full method, optimisation record and caveats: [docs/evaluation.md](docs/evaluation.md).

**Retrieval** — corpus 21,910 chunks / 16 documents; production configuration (dense 0.75 + BM25 + exact-clause leg → RRF → cross-encoder over the top 12 → guard → expansion); k_eval = 20; 2026-09-13.

| Dataset | Pipeline leg | R@5 | R@10 | R@20 | Hit@5 | MRR | nDCG@10 | RegHit@5 |
|---|---|---|---|---|---|---|---|---|
| `regulatory_v1` (47 human-written) | dense only | 0.546 | 0.639 | 0.677 | 0.667 | 0.470 | 0.480 | 0.864 |
| | sparse only (BM25) | 0.686 | 0.821 | 0.870 | 0.795 | 0.546 | 0.582 | 0.955 |
| | hybrid RRF | 0.650 | 0.851 | 0.875 | 0.769 | 0.583 | 0.615 | 0.955 |
| | **full** | **0.844** | **0.877** | **0.926** | **0.949** | **0.756** | **0.768** | **0.977** |
| `regulatory_v2` (262) | dense only | 0.608 | 0.660 | 0.703 | 0.638 | 0.480 | 0.512 | 0.871 |
| | sparse only (BM25) | 0.896 | 0.930 | 0.961 | 0.926 | 0.720 | 0.761 | 0.988 |
| | hybrid RRF | 0.818 | 0.916 | 0.949 | 0.848 | 0.652 | 0.704 | 0.984 |
| | **full** | **0.911** | **0.931** | **0.957** | **0.938** | **0.808** | **0.829** | **0.992** |

Against the pre-v2 baseline (heuristic reranker, equal RRF weights) the full pipeline moved from MRR 0.636 → 0.756 on v1 and 0.709 → 0.808 on v2. Retrieval p50 is ~1.4 s on a laptop CPU with the cross-encoder (270 ms with `RERANKER=heuristic`). The regression gate (`tests/retrieval_regression`) fails CI below MRR 0.60 (v1) / 0.68 and R@10 0.90 (v2) or when any of 23 stable cases loses its clause.

**End-to-end answers with a real LLM** — `regulatory_v2`, 262 cases, free-tier models through an OpenAI-compatible gateway (`LLM_MODEL=auto`), 2026-09-13:

| Metric | Value |
|---|---|
| refusal accuracy (unanswerable → abstain, answerable → answer) | 0.943 — 11/11 not-in-corpus/out-of-scope abstained; false refusals 5.7 % |
| citation hit / precision (expected regulation + clause) | 0.919 / 0.714 |
| key-fact coverage in the answer / in the retrieved evidence | 0.823 / 0.977 |
| grounding validator accepted the draft | 0.965 |
| adversarial injection resisted (6 cases) | 6/6 |
| RAGAS (n = 100): faithfulness · answer relevancy · context precision · context recall | 0.742 · 0.774 · 0.866 · 0.960 |
| DeepEval (n = 26): faithfulness · answer relevancy · contextual precision | 1.000 · 0.907 · 0.880 |

**Ingestion** — 16 registry sources, 11,700 sections (275 typed definitions), 847 cross-references, 9,022 tables, 4,120 figures; re-ingesting an unchanged source is a no-op; uploads run through the same pipeline via the queue worker (upload → READY for a 3-page note in ~10 s locally).

**Product flows** — Playwright, real API + worker + LLM: login → ask → evidence · upload → READY → ask the document · logout → login → restore · user A private upload → user B denied · failed upload → actionable error: 5/5.

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
   FastAPI (identity, workspaces, conversations, documents, jobs; RBAC scopes, rate limit, request ids, OTel, Prometheus)
                                  │                                   ▲
   worker: ingestion_jobs (SKIP LOCKED) → same stage pipeline         │  Next.js workbench: /app/{home,chat,documents,upload,ingestion,settings,admin}
```

Authorization is one predicate (`retrieval/authz.py`) — organization membership for verified sources, workspace membership for workspace documents, ownership for private documents — evaluated in SQL before ranking and mirrored in the BM25 pre-filter; a unit test proves both evaluators agree. Full picture: [docs/architecture.md](docs/architecture.md), [ADR-0029](docs/ADR/0029-v2-product-domain-schema-and-authorization.md).

The bounded agent (`agents/graph.py`, LangGraph) adds two routes on top of the standard path: **comparison** (one scoped retrieval per named regulation) and **change analysis** (section-level diff between the two latest versions as data for the model). Budgets — retrieval attempts, LLM calls, tool calls, wall-clock — are enforced in code.

## Why these decisions

| Decision | Why | ADR |
|---|---|---|
| PostgreSQL + pgvector as the only store | one system of record for metadata, audit and vectors; HNSW meets the measured latency; no second database to keep consistent | [0019](docs/ADR/0019-canonical-store-and-schema-cutover.md) |
| Structural chunks (clause tree), not fixed windows | citations must name exact clauses; merged tiny siblings keep numbers inline; tables carry headers | [0020](docs/ADR/0020-structural-chunking.md) |
| Dense + BM25 with reciprocal-rank fusion | measured: RRF beats either leg alone (MRR 0.576 vs 0.523/0.546); ranks fuse, raw scores are never summed | [0021](docs/ADR/0021-hybrid-retrieval-and-rrf.md) |
| Cross-encoder reranker over the top-12 fused candidates | measured on 262+47 cases: MRR +0.10 to +0.12 over the heuristic for ~1.2 s on CPU; heuristic remains a config switch | [0022](docs/ADR/0022-reranking.md), [docs/evaluation.md](docs/evaluation.md) |
| Versions with validity windows + lifecycle states | "latest" means latest *in force*, never latest downloaded; historical queries filter before ranking | [0023](docs/ADR/0023-temporal-regulation-model.md) |
| Citation contract with programmatic validation | the model may not invent ids, pages or numbers; violations are dropped, not trusted | [0024](docs/ADR/0024-citation-contract.md) |
| Bounded LangGraph, no swarm | routing/decomposition is deterministic; the LLM only synthesises under schema | [0025](docs/ADR/0025-bounded-agent.md) |
| Typed provider interfaces, fakes only in `test` | production refuses hashing embeddings, mock LLM, anonymous auth at startup | [0026](docs/ADR/0026-provider-abstraction-and-fakes.md) |
| Explicit data-class policy per LLM provider | confidential evidence never reaches a provider that is not cleared; policy is config, not a model-name heuristic | [0027](docs/ADR/0027-security-policy.md) |
| ECS Fargate + RDS + S3, no Kubernetes | one stateless service and two managed stores do not justify a cluster | [0028](docs/ADR/0028-deployment-architecture.md) |
| Identity/workspace tables + one document-level authorization predicate; PostgreSQL-backed job queue; conversations as continuity, never evidence | no new services; predicate before ranking; uploads can never leak or silently become authoritative | [0029](docs/ADR/0029-v2-product-domain-schema-and-authorization.md) |

## Ingestion, versioning, freshness

- **Registry** `knowledge/00_registry/sources.yaml` is the allowlist: regulation identity, kind, jurisdiction, authority level, data class, official URI, SHA-256/size, and the consolidated text's own version label, series, revision, publication and entry-into-force dates (human-reviewed; the cover-page parser cross-checks them and logs disagreements).
- **Idempotency keys**: parse = `source_sha256 + parser_version + parser_config_hash`; chunk = `parsed_hash + chunker_version + chunker_config_hash`; embed = `chunk_sha256 + model_version + dimensions`; index = `chunk_id + index_schema_version`. Unchanged content is never re-embedded — chunk content is version-independent so an amendment reuses every untouched clause's vector.
- **Lifecycle** is a state machine (`domain/regulations/lifecycle.py`); only `ACTIVE` versions are retrievable for current queries, `ACTIVE|SUPERSEDED` for as-of queries. Activation supersedes the previous version in the same transaction and closes its validity window. Bad files are quarantined with an attempt budget; every transition is an `ingestion_events` row.
- **Freshness**: `sa_freshness_lag_days{regulation}` (publication → activation) and `ingestion_runs.stats`.

## Retrieval pipeline

Details in [docs/retrieval-design.md](docs/retrieval-design.md). Highlights: SQL scope (status, validity, regulation, data class) before any ranking; exact identifiers (`5.2.1.8`, `Annex 3`) route to a dedicated leg with weight 2 in RRF; BM25 is a process-cached index with in-memory scope filtering; the heuristic reranker rewards literal overlap, authority and normative clauses for requirement-seeking questions; the diversification cap is scope-aware; parent sections and resolved cross-references are attached to each evidence item within a token budget.

## Evaluation

Two gold sets: `regulatory_v1` (47 human-written cases, 16 slices) and `regulatory_v2` (262: v1 + 200 cases generated from section text with verbatim-verified key facts + hand-written unanswerable/ambiguous/adversarial cases). Retrieval legs are measured independently (`make eval`), configurations on a grid (`scripts/eval/grid.py`), and the end-to-end pipeline with deterministic answer metrics plus RAGAS and DeepEval judges through the same LLM gateway (`make eval-judged`). Method, results and the optimisation record: [docs/evaluation.md](docs/evaluation.md).

## Security and governance

Organization roles (engineer, knowledge_admin, auditor, org_admin) map to scopes on every route; browser sessions are HttpOnly cookies with a CSRF header requirement, API keys/OIDC for machines; the document-level authorization predicate and data classes narrow the retrieval universe *before* ranking (cross-user, cross-workspace and cross-organization isolation tested at listing, detail, job, evidence and retrieval level); uploads are bounded and validated, quarantined files never index, promotion to the verified corpus is privileged and audited; SSRF-safe fetcher (https, allowlist, public-IP DNS check per hop, size cap, conditional GET); file validation (magic bytes, size, page count, hash against the registry); prompt-injection signals recorded and never executed; explicit LLM data-class policy; per-principal rate limit; audit actor on privileged ingestion; secrets from the environment/secret manager only. Threat model: [docs/security-threat-model.md](docs/security-threat-model.md).

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
uv run safety-assistant users add --email you@example.com --name "You" --role engineer --workspace default
uv run uvicorn safety_assistant.api.main:app --port 8010        # make api
uv run safety-assistant worker                                  # make worker — processes uploads
cd frontend && npm install && npm run dev                       # http://localhost:3010 → sign in with the seeded email
```

`.env.example` enables dev login (`DEV_LOGIN_ENABLED=true`) and sets a development `SESSION_SECRET`; configure `LLM_*` for generated answers (without an LLM the system answers in evidence-only mode). Without the licensed corpus, `uv run python scripts/maintenance/seed_synthetic_corpus.py` ingests the synthetic two-version regulation used by the tests (CI does this for the Playwright job).

Quality gates: `make lint types test` (156 tests; PostgreSQL required for integration/e2e), `make eval`, `make eval-judged` (needs an LLM), `make frontend`, `make e2e` (API + worker running), `make load`.

## Deployment

- Image: `infra/docker/Dockerfile` (multi-stage uv build, non-root, read-only root fs, embedding model baked in, healthcheck). Full local stack with MinIO: `docker compose -f infra/docker/compose.yaml up --build`.
- CI: `.github/workflows/ci.yml` (ruff, mypy, all test tiers incl. migration round-trip, frontend type-check/lint/build + Playwright flows against a seeded synthetic corpus, container build, Trivy, SBOM), `security.yml` (pip-audit, bandit, gitleaks, semgrep), `eval.yml` (nightly retrieval evaluation; real corpus restored from a bucket when configured), `release.yml` (ghcr image with provenance + SBOM on tags).
- Infrastructure: `infra/terraform/` — ALB (TLS 1.3) → ECS Fargate service with circuit-breaker rollback → RDS PostgreSQL 16 (TLS forced, encrypted, 14-day backups/PITR, multi-AZ in production) + versioned, encrypted S3 artifacts + Secrets Manager + CloudWatch alarms. Container: `APP_ENV=production`, `AUTH_MODE=oidc`, `SESSION_SECRET` ≥ 32 chars, `DEV_LOGIN_ENABLED=false` (enforced at startup). Run the worker as a second service (`entrypoint.sh worker`) — `infra/docker/compose.yaml` shows the topology; the Terraform module does not yet define the worker task.

## Known limitations

- **LLM quality depends on free-tier routing.** Answers and judges ran through a gateway that routes to different free models per request; the routed model is recorded per answer but results are not attributable to one model, and p50 latency (9 s) is dominated by routing/rate-limit retries. Point `LLM_*` at one dependable model before production use; `LLM_DATA_CLASSES` keeps confidential evidence away from providers that are not cleared.
- **RAGAS faithfulness (0.74) is a lower bound**: attribution sentences count against it; the numeric/citation validator (0.965) and DeepEval (1.00, n = 26) judge the requirement claims. DeepEval covered 26 of a planned 40 records because a gateway call hung; the sample is what completed, not a selection.
- **200 of the 262 v2 cases are LLM-generated** (facts verified verbatim, not human-reviewed). Sparse-heavy tuning that helped them regressed the human-written set, which is why v1 remains the tie-breaker.
- **Cross-encoder latency** (~1.2 s per query on CPU) applies to `/search` too; use `RERANKER=heuristic` where interactive search latency matters more than MRR.
- **One version per regulation in the corpus**; temporal behaviour is proven on a synthetic two-version regulation.
- **Parser**: PyMuPDF only; scanned pages are flagged, not OCR'd; no malware scanner on uploads (the boundary is `documents.service.create_upload`).
- **Per-process rate limiter and BM25 index**; the queue is a PostgreSQL table polled by workers (`ponytail:` a broker behind the same `enqueue()`/`run_once()` seam is the upgrade path).
- **OIDC** is verified against JWKS in tests, not a live IdP; the browser OIDC redirect is not implemented (dev login and API keys are). Dark mode is not implemented.
- **Terraform is unapplied** (no cloud account) and does not yet define the worker task; Trivy/SBOM/pip-audit/gitleaks/semgrep run in CI, not locally. Load figures are from the pre-cross-encoder configuration.
