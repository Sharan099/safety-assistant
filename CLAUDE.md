# CLAUDE.md — Passive Safety CAE Investigation Agent

This file is the standing instruction set for Claude Code in this repository.
Read the design contracts before touching code — they are authoritative and
supersede improvisation:

- `PRD.md` — product requirements
- `TRD.md` — technical requirements
- `APP_FLOW.md` — application flow / states
- `BACKEND_SCHEMA.md` — database schema
- `UI_UX_DESIGN_BRIEF.md` — UI contract
- `IMPLEMENTATION_PLAN.md` — phased build order (the source of truth for "what's next")
- `ENVIRONMENT_SETUP.md` — tooling, dependencies, knowledge architecture
- `CLAUDE_CODE_BOOTSTRAP_PROMPT.md` — the original bootstrap instructions this repo was built from

## Golden rule

Do not build the agent or the UI first. Build in this order (see
`IMPLEMENTATION_PLAN.md` for full phase list):

```
environment → repository skeleton → domain schema → source registry →
document ingestion → deterministic analysis → synthetic benchmark →
evidence/provenance → RAG → LLM provider abstraction → LangGraph agent →
UI → production hardening
```

The deterministic engineering layer (`packages/analysis`) must work with
zero LLM calls. Never let it silently depend on one.

## Non-negotiable guardrails (PRD §8, TRD §25)

The AI layer must never, silently:
- calculate numerical engineering metrics itself (that's `packages/analysis`);
- invent solver behavior, regulatory limits, page numbers, or evidence;
- claim a document was retrieved when it wasn't;
- claim causality from temporal precedence alone;
- override a quality-gate failure or engineer review;
- fill a knowledge gap with fabricated content — represent it as
  `NOT_AVAILABLE` / `NOT_AUTHORIZED` / `NOT_INGESTED` instead.

"Unknown" is a distinct, first-class value from "pass" everywhere in this
system (quality, comparability, config diff, etc.) — never collapse it.

## Development workflow: Ponytail

[Ponytail](https://github.com/DietrichGebert/ponytail) is installed as a
user-scope Claude Code plugin (`claude plugin marketplace add
DietrichGebert/ponytail && claude plugin install ponytail@ponytail`, done
2026-08-12 — see `docs/ADR/0002-ponytail-dev-workflow.md`). It is a
development-time discipline only, never a runtime dependency of the
application.

Its decision ladder applies to every change in this repo, in order:
1. Does this need to exist at all?
2. Is it already in the codebase — reuse it.
3. Is it in the standard library — use it.
4. Is it a native platform/framework feature — use it.
5. Is it in an already-installed dependency — use it.
6. Can it be one line?
7. Only then write the minimum implementation.

Prefer `/ponytail-review` on a diff before considering a change done.

## Runtime LLM: FreeLLMAPI

FreeLLMAPI (https://github.com/tashfeenahmed/freellmapi) is an OpenAI-compatible
local proxy (default `http://localhost:3001/v1`) aggregating free-tier
providers. Treat it as unreliable by design (no SLA, no frontier models,
provider quotas rotate). Always go through the `LLMProvider` abstraction
(`FreeLLMAPIProvider` / `MockProvider`); never hardcode a specific model —
configure via `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`. See
`docs/ADR/0003-freellmapi-runtime-llm.md`.

Claude Sonnet (via Claude Code) is the **development/coding** model. FreeLLMAPI
is the **application runtime** model. Never conflate the two.

## Environment

- Python: `uv` only (no Conda, no global pip). `uv run <cmd>` for everything.
- Node: Next.js + TypeScript + Tailwind + TanStack Query (frontend, Phase 17+).
- DB: PostgreSQL + pgvector (`docker compose up postgres`). No separate vector DB
  unless a measured benchmark proves pgvector inadequate.
- Large numerical/time-series data: Parquet + DuckDB, never raw rows in Postgres.
- Target dev hardware is an 8 GB RAM / 2 GB GPU laptop — keep services minimal,
  avoid local large-model inference, process the knowledge corpus incrementally.

## Knowledge corpus rules (PRD §6-7, TRD §10-12)

- Only documents listed in `knowledge/00_registry/source_manifest.yaml` may be
  treated as available knowledge.
- Original PDFs in `Knowledge source/` are immutable and never modified;
  canonical copies live under `knowledge/01_regulations/` and
  `knowledge/02_official_docs/`.
- Proprietary/licensed PDFs are never committed to Git (see `.gitignore`).
  Only the manifest, schemas, and generated OKF markdown are versioned.
- Source authority is never flattened: `REGULATION > OFFICIAL_DOCUMENTATION >
  INTERNAL_APPROVED > HISTORICAL > SYNTHETIC > LLM_REASONING`.
- The public PAM-CRASH spec sheet is `REFERENCE`, never a substitute for a
  licensed PAM-CRASH manual.

## Definition of done (IMPLEMENTATION_PLAN §25)

A feature is complete only when it has: implementation + tests + error
handling + logging + documentation + evaluation + provenance. Do not move to
the next phase if a gate fails — report the failure instead.

## Commands

```powershell
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy .              # types
docker compose up -d postgres
```
