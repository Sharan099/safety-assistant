# Passive Safety CAE Investigation Agent

An engineering investigation workstation for passive-safety / occupant-protection
CAE engineers — not a chatbot. It helps answer questions like *"why did chest
deflection increase between Run A and Run B?"* with evidence-backed, traceable
analysis: quality gates, comparability checks, configuration diffs, signal
analysis, divergence detection, and citation-grounded regulatory/solver
retrieval, with an engineer review step before any conclusion is recorded.

## Status

**V1 vertical slice complete and working end to end**, backend and UI:
select two runs → quality gate → comparability → configuration diff →
signal analysis/divergence → evidence → agent-drafted hypothesis → engineer
review — all real, tested, and exercised against the real UN_R94/LS-DYNA
knowledge corpus and the SCN-001..SCN-010 synthetic benchmark. See
`IMPLEMENTATION_PLAN.md` for the full phased build order and `docs/ADR/` for
decisions made along the way (10 ADRs so far).

Not yet built: real historical-case data (the retrieval code path exists and
is honestly empty until investigations are closed), a chosen/benchmarked
embedding model (an interim deterministic placeholder is in place, see
`docs/ADR/0007`), report generation, mechanism/animation review, and
production hardening (auth, audit trail, a dedicated test database).

## Start here

Read these in order before changing anything — they are the product/technical
contract, not background reading:

1. `PRD.md` — what this is and why
2. `TRD.md` — how it's built
3. `APP_FLOW.md` — the investigation flow and states
4. `BACKEND_SCHEMA.md` — the data model
5. `UI_UX_DESIGN_BRIEF.md` — the UI contract
6. `IMPLEMENTATION_PLAN.md` — phase-by-phase execution order
7. `ENVIRONMENT_SETUP.md` — tooling and knowledge-ingestion architecture
8. `CLAUDE.md` — standing rules for AI-assisted development in this repo

## Repository layout

```
apps/api             FastAPI backend — runs, investigations, knowledge search
apps/web              Next.js investigation workspace (port 3010)
packages/domain        Core entities (43 tables) + Alembic migrations
packages/analysis        Deterministic, LLM-free CAE analysis (no LLM calls)
packages/ingestion         PyMuPDF-based knowledge ingestion pipeline
packages/retrieval           RAG: Postgres FTS + pgvector + RRF
packages/agent                  LLMProvider + the LangGraph investigation agent
knowledge/            Canonical knowledge corpus + registry + OKF concepts
data/                 Synthetic runs, Parquet signal data (gitignored, regenerable)
evals/                Golden datasets and evaluation harness (not started)
docs/ADR/             Architecture decision records
```

## Getting started

```powershell
uv sync                                          # install Python deps
docker compose up -d postgres                    # port 5433 — see docs/ADR/0004
uv run alembic upgrade head                      # apply the domain schema
uv run python scripts/generate_synthetic_dataset.py   # SCN-001..010 -> Postgres + Parquet
uv run python scripts/ingest_documents.py         # UN_R94 (full) + 2 LS-DYNA manuals (60p)
uv run python scripts/index_knowledge.py          # embed ingested chunks
uv run uvicorn apps.api.main:app --port 8010      # backend, port 8010 (docs/ADR/0004)

# in another terminal:
cd apps/web
cp .env.local.example .env.local
npm install
npm run dev                                       # frontend, http://localhost:3010
```

```powershell
uv run pytest              # 67 tests (1 opt-in destructive migration test skipped by default)
uv run ruff check .        # lint
uv run mypy apps packages scripts tests   # strict type check
```

Copy `.env.example` to `.env` and fill in secrets before running anything
that talks to a database or LLM provider. Never commit `.env`.

## Knowledge corpus

Source PDFs live in `Knowledge source/` (local, untouched originals) and are
registered with SHA-256 hashes in `knowledge/00_registry/source_manifest.yaml`.
See that manifest for exactly which documents are available and their
authority level — the system must never fabricate content for documents that
aren't listed there.
