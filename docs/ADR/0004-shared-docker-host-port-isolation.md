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
