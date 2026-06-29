# Passive Safety Assistant — Product Requirements Document

**Version:** 1.0 (Final consolidation)  
**Date:** 2026-06-27  
**Audience:** EU passive-safety engineers, program managers, homologation teams

---

## 1. Purpose

The Passive Safety Assistant is a retrieval-augmented chat system that answers engineering questions about **EU passive-safety regulations** (primarily UNECE WP.29 / GRSP scope) with **grounded citations** from an official document corpus.

It complements the Regulation Knowledge-Base Registry (see `Regulation_KnowledgeBase_Registry_PRD.md`) which owns acquisition, validation, chunking, and embedding.

---

## 2. Users & jobs-to-be-done

| User | Job |
|------|-----|
| Crash/safety engineer | Look up injury criteria, test conditions, anchorage tables across UN R94/R95/R14/R16/etc. |
| Homologation engineer | Confirm which regulation clause applies to a vehicle category |
| Program manager | Compare legacy vs current child-restraint requirements (R44 vs R129) |

---

## 3. Functional requirements

### FR-1 Chat interface
- Single-page web UI: question input, streamed/static answer, citation sidebar `[S1]…`
- Shows serving model, per-step latency, evidence-only state when LLM unavailable

### FR-2 Grounded answers
- Every factual claim must trace to retrieved regulation chunks
- Citations include regulation code, document, section/clause, page

### FR-3 Hybrid retrieval
- Dense (Nomic 768-d) + sparse keyword + RRF fusion + cross-encoder rerank (when not on Windows mock path)
- Query expansion for R94 chest/ThCC, R14 Annex 6 ISOFIX tables, multi-reg comparisons

### FR-4 LLM gateway
- Failover chain: `groq (70B) → groq_fast (20B) → openrouter_llama → openrouter_claude → evidence-only`
- 3s connect timeout, circuit breaker on repeated connection failures
- No Llama 3.1 8B in live chain (decommissioned on Groq)
- Direct Anthropic omitted when firewall-blocked; Claude via OpenRouter when key set

### FR-5 Readiness gate
- `GET /api/v1/ready` runs lightweight retrieval probe; UI disabled until ready

### FR-6 Observability
- Per-request timing: guardrails, dense, sparse, RRF, rerank, annex promotion, parent expansion, LLM generation
- `GET /api/v1/coverage` for Europe passive-safety gap report

---

## 4. Non-functional requirements

| NFR | Target |
|-----|--------|
| Corpus embedding | `nomic-ai/nomic-embed-text-v1.5` @ 768-d (pinned) |
| Answer latency (p50) | < 15s with warm embedder (excluding first cold start) |
| Availability | Evidence-only fallback must never blank-error |
| Scope | Europe passive safety only per `coverage_expected.yaml` |

---

## 5. Out of scope

- Active safety, lighting, braking, emissions
- FMVSS / IIHS / China NCAP (excluded from EU corpus)
- Authoritative Ragas judge on firewall-blocked networks (runbook: `output/RUN_AUTHORITATIVE_ELSEWHERE.md`)

---

## 6. API surface

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/health` | DB + Redis + Celery liveness |
| GET | `/api/v1/ready` | Chat enable gate |
| POST | `/api/v1/chat` | Question → answer + citations + routing |
| POST | `/api/v1/search` | Raw retrieval + synthesis |
| GET | `/api/v1/coverage` | Expected vs ingested gap report |

---

## 7. Environment variables

| Variable | Required | Notes |
|----------|----------|-------|
| `GROQ_API_KEY` | Recommended | Primary 70B tier |
| `OPENROUTER_API_KEY` | Recommended | Failover when Groq TPD/rate limited (+5.5% fee logged) |
| `ENABLE_GATEWAY` | Default `true` | Multi-tier routing |
| `STRUCTURE_CHUNKING` | Default `false` | Set `true` for clause/table-aware chunks |
| `USE_MOCK_EMBEDDINGS` | Windows dev | Set `false` + `EMBED_BACKEND=transformers` for real Nomic |
| `NEXT_PUBLIC_API_URL` | Frontend | e.g. `http://127.0.0.1:8000/api/v1` |

---

## 8. Success criteria (demo)

1. Engineer opens UI, asks R94 chest limit → cited answer with ThCC/42 mm sources in top-8
2. R14 Annex 6 ISOFIX table question → M1/M2/N1 grid values cited
3. R29 cab protection question → sources from newly ingested UN R29 (post Phase 2)
4. When Groq cap hit → OpenRouter or evidence-only with honest labeling

---

## 9. Related documents

- `Regulation_KnowledgeBase_Registry_PRD.md` — acquisition pipeline PRD
- `coverage_expected.yaml` — expected EU passive-safety coverage
- `docs/EMBEDDING_EVIDENCE.md` — Nomic 768-d pinning evidence
- `output/FINAL_CONSOLIDATION_LOG.md` — consolidation run log
