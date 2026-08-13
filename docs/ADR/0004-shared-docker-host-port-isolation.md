# ADR-0004: Port isolation from sibling projects on the shared Docker host

- **Status:** Accepted
- **Date:** 2026-08-12

## Context

This dev machine already runs Docker containers from sibling/unrelated
projects (`docker ps -a` at Phase-2 start showed an `autosafety_rag` stack:
`pgvector/pgvector:pg16` on host port 5432, a `qdrant` instance, a
`portkey` gateway, and — notably — a `ghcr.io/tashfeenahmed/freellmapi:latest`
container already **running and healthy** on `127.0.0.1:3001`).

## Decision

1. This project's `docker-compose.yml` publishes Postgres on host port
   **5433**, not 5432, and uses an explicit `name:`/`container_name:` so
   `docker compose up/down` here never touches the sibling stack.
2. FreeLLMAPI is treated as already available at the ADR-0003 default
   (`http://localhost:3001/v1`) — it does not need to be installed by this
   project; it's shared host infrastructure. `.env.example` still exposes
   `LLM_BASE_URL` so this remains configurable, not assumed.
3. No code/config in this repo references Qdrant or Portkey — per ADR-0001,
   this project does not use a separate vector DB, and Portkey is unrelated
   to the `LLMProvider` abstraction here.

## Consequences

- Running `docker compose up -d postgres` in this repo is safe regardless of
  whether the sibling stack is up.
- `DATABASE_URL` in `.env.example` uses port 5433.
- If FreeLLMAPI is ever stopped/removed from the shared host, `LLMProvider`
  must fail gracefully per TRD §30 (preserve state, surface "unavailable",
  allow retry) rather than assume it's always reachable.

## Update — 2026-08-13, Phase 17-18 (non-Docker ports)

The same problem exists outside Docker: this host also runs
`H:\AutoSafety_RAG`'s own FastAPI backend on `127.0.0.1:8000` (with its own
security-header middleware — confirmed by curling it and seeing CSP/HSTS
headers this project's API never sets) and its Next.js frontend on
`0.0.0.0:3000`. Both are extremely common defaults, so this project
deliberately avoids them too:

- `apps/api` (FastAPI/uvicorn): port **8010** — not committed to a script
  default anywhere; run explicitly with `uv run uvicorn apps.api.main:app
  --port 8010`.
- `apps/web` (Next.js): port **3010** — set directly in `package.json`'s
  `dev`/`start` scripts, so `npm run dev` never needs the flag repeated.
- `apps/web/.env.local.example`'s `NEXT_PUBLIC_API_URL` points at
  `http://localhost:8010/api/v1` accordingly.

General rule for this repo: before hardcoding *any* "default" port, check
`netstat -ano` on this host first — assume common ports (3000, 5432, 8000,
8080, ...) are taken by sibling projects until proven otherwise.
