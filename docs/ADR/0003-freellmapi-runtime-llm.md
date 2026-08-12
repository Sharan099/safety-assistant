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

## Consequences

- Given FreeLLMAPI's explicit "no SLA" stance, `packages/analysis` (the
  deterministic layer) must remain fully usable with the LLM unavailable
  (TRD §30, APP_FLOW §21) — this is a hard requirement, not best-effort.
- FreeLLMAPI itself is not installed/run as part of this repo's setup; it is
  an external local service the developer starts separately. Actual
  installation is deferred to Phase 14 (`IMPLEMENTATION_PLAN.md`) when the
  provider is first wired up.
- Claude Sonnet (via Claude Code) remains strictly the coding/development
  model and is never conflated with the application's runtime LLM.
