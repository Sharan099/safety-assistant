# ADR-0026 — Typed provider interfaces; fakes only in the test profile

Status: accepted · Date: 2026-09-11 · Supersedes: 0003, 0007, 0011, 0014 (kept as history)

## Decision
`EmbeddingProvider`, `Reranker`, `LLMProvider` are Protocols under `providers/`. Real implementations: fastembed (ONNX) embeddings and cross-encoder; an OpenAI-compatible chat client with 429/5xx/4xx classification, bounded jittered retry and JSON-schema output validated in code. `HashingEmbeddingProvider` and `MockLLMProvider` exist only for `APP_ENV=test`; `Settings` refuses them in production together with `AUTH_MODE=none` and the development database password. Readiness fails if the embedding model cannot load; an LLM outage is a degradation, not a readiness failure. Provider calls are synchronous and run in the API threadpool (CPU-bound ONNX work would otherwise block the event loop — verified by test).

## Evidence
`tests/unit/test_providers_and_config.py`, `tests/integration/test_fault_injection.py`.
