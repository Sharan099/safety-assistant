# ADR-0003: FreeLLMAPI as the initial runtime LLM, behind a provider abstraction

- **Status:** Accepted
- **Date:** 2026-08-12

## Context

`PRD.md` §9 and `TRD.md` §4 require a configurable, swappable runtime LLM
gateway, with [tashfeenahmed/freellmapi](https://github.com/tashfeenahmed/freellmapi)
as the first option, without hardcoding a specific model or assuming
availability.

## Investigation

FreeLLMAPI is a self-hosted OpenAI-compatible proxy aggregating ~29 free-tier
LLM providers behind one API surface:

- `POST /v1/chat/completions` (+ `/v1/completions`, `/v1/embeddings`,
  `/v1/images/generations`, `/v1/audio/speech`, `/v1/models`), plus an
  Anthropic-compatible `/v1/messages` surface.
- Single bearer token (`freellmapi-<key>`); per-provider credentials encrypted
  at rest.
- Runs locally via `curl -fsSL https://freellmapi.co/install.sh | bash`, Docker,
  or native install; Node 20+; dashboard on `http://localhost:3001`.
- Explicit caveats from the project itself: **no SLA, no frontier models,
  variable latency, daily free-tier caps reset at UTC midnight** — "personal
  experimentation and learning, not production."

## Decision

Model the runtime LLM behind an `LLMProvider` interface:

```
LLMProvider
├── FreeLLMAPIProvider   (OpenAI-compatible client → LLM_BASE_URL, default
│                          http://localhost:3001/v1)
└── MockProvider          (deterministic responses for tests / offline dev)
```

Configuration is env-var driven only (`LLM_PROVIDER`, `LLM_BASE_URL`,
`LLM_MODEL`, `LLM_API_KEY` — see `.env.example`). No model name is hardcoded
anywhere in application code.

## Update — 2026-08-13, Phase 14 implementation

`packages/agent/llm.py` implements the `LLMProvider`/`MockProvider`/
`FreeLLMAPIProvider` design above. Confirmed live against the actual
FreeLLMAPI container running on this host (docs/ADR/0004): an
unauthenticated request to `/v1/models` returns `401 {"error": {"message":
"Invalid API key"}}` — no credential for it is available in this
environment. `FreeLLMAPIProvider.complete()` was exercised against that
live endpoint and correctly raised `LLMUnavailableError` rather than
hanging, retrying silently, or crashing uncaught — the TRD.md Section 30
failure path this ADR requires. Once a real `freellmapi-<key>` is available,
set `LLM_API_KEY`/`LLM_MODEL` in `.env` and no code changes are needed.

## Consequences

- Given FreeLLMAPI's explicit "no SLA" stance, `packages/analysis` (the
  deterministic layer) must remain fully usable with the LLM unavailable
  (TRD §30, APP_FLOW §21) — this is a hard requirement, not best-effort.
- FreeLLMAPI itself is not installed/run as part of this repo's setup; it is
  an external local service the developer starts separately. Actual
  installation is deferred to Phase 14 (`IMPLEMENTATION_PLAN.md`) when the
  provider is first wired up.
- The coding-assistant model used during development is a separate concern
  from the application's runtime LLM and must never be conflated with it.
