# Passive Safety CAE Investigation Agent

An engineering investigation workstation for passive-safety / occupant-protection
CAE engineers — not a chatbot. It helps answer questions like *"why did chest
deflection increase between Run A and Run B?"* with evidence-backed, traceable
analysis: quality gates, comparability checks, configuration diffs, signal
analysis, divergence detection, and citation-grounded regulatory/solver
retrieval, with an engineer review step before any conclusion is recorded.

## Status

**V1, Phase 1 (repository foundation).** No agent, no UI, no ingestion pipeline
yet — by design. See `IMPLEMENTATION_PLAN.md` for the full phased build order
and `docs/ADR/` for decisions made along the way.

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
apps/api          FastAPI backend (not yet implemented)
apps/web           Next.js frontend (not yet implemented)
packages/domain     Core entities (Project, Run, Signal, ...)
packages/analysis    Deterministic, LLM-free CAE analysis
packages/ingestion   Knowledge-source ingestion pipeline
packages/retrieval   RAG (Postgres FTS + pgvector + RRF)
packages/agent       LangGraph investigation agent
knowledge/           Canonical knowledge corpus + registry + OKF concepts
data/                Synthetic runs, artifacts, Parquet signal data
evals/               Golden datasets and evaluation harness
docs/ADR/            Architecture decision records
```

## Getting started

```powershell
uv sync                     # install Python deps
docker compose up -d postgres
uv run pytest                # run tests (Phase-0 smoke + manifest checks)
uv run ruff check .          # lint
```

Copy `.env.example` to `.env` and fill in secrets before running anything
that talks to a database or LLM provider. Never commit `.env`.

## Knowledge corpus

Source PDFs live in `Knowledge source/` (local, untouched originals) and are
registered with SHA-256 hashes in `knowledge/00_registry/source_manifest.yaml`.
See that manifest for exactly which documents are available and their
authority level — the system must never fabricate content for documents that
aren't listed there.
