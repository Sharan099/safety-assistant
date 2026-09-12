# ADR-0025 — Bounded LangGraph orchestration, no multi-agent swarm

Status: accepted · Date: 2026-09-12 · Supersedes: the pre-rebuild investigation/copilot graphs

## Decision
One typed graph: parse_query → route_intent → {standard | comparison | change_analysis} → evidence_gate → (one rewrite) → generate → citation_validate. Routing, scope, dates and budgets are deterministic; the LLM is called at most once and only under schema. Tools are narrow and typed (`search_regulations`, `get_regulation_versions`, `compare_versions`, `get_section`) — no shell, SQL strings or network. `Budget` caps retrieval attempts (3), LLM calls (1), tool calls (8) and wall clock (45 s); exhaustion abstains.

## Alternatives rejected
Multi-agent swarms, MCP tool servers, GraphRAG: no evaluation slice justified them; they remain experiments.

## Evidence
`tests/integration/test_agent_routes.py` (routes, budgets), `tests/integration/test_api_ask.py`.
