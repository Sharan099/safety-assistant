# ADR-0027 — Security policy: allowlists, RBAC before retrieval, LLM data-class policy, no shared answer cache

Status: accepted · Date: 2026-09-12

## Decision
- Sources: registry allowlist with SHA-256; SSRF-safe fetcher (https only, allowlisted hosts, public-IP DNS check per redirect hop, streaming size cap, PDF content type, conditional GET); file validation (magic, size, page count, hash).
- Identity: `api_key` (dev/CI) or `oidc` (JWKS, audience/issuer/expiry) → roles `RegulationViewer | Engineer | DataIngestor | Auditor | SafetyAdmin` → scopes. Scopes gate every route; the principal's data classes narrow `ScopeFilter` **before** ranking.
- LLM policy: `LLM_DATA_CLASSES` declares what the provider may see; confidential evidence with a PUBLIC-only provider yields evidence-only mode. Never a model-name heuristic.
- Prompt injection: signals are detected and recorded; evidence and questions are data by contract; validation drops unsupported claims.
- Caching: there is **no shared answer cache**. Retrieval p50 is ~270 ms without one; a cache would need tenant/data-class/prompt/index-version keys and encryption for confidential answers — not worth it until measured. The only in-process cache is the BM25 index (no user data).
- Audit: privileged ingestion records the actor; every query writes a trace; secrets come from the environment/secret manager.

## Evidence
`tests/security/` (30 tests); `docs/security-threat-model.md`.
