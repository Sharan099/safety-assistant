# Automotive Safety RAG — Passive Safety Assistant

EU passive-safety regulation knowledge base with hybrid retrieval, multi-tier LLM gateway failover, structure-aware chunking, and a demo chat UI for engineers.

## What it does

- Ingests official UNECE passive-safety PDFs through a **gated pipeline** (allowlist → validate → storage → structure chunk → embed)
- Answers engineering questions with **grounded citations** (`[S1]`… from regulation, clause, page)
- Reports **coverage gaps** vs `coverage_expected.yaml` (22 UNECE regs, Europe-only scope)
- Routes LLM calls through **Groq → OpenRouter → evidence-only** when capacity or network blocks generation
- Supports **harness / confidential test-record** ingest for internal validation data (separate from public UNECE corpus)

## Architecture

```
┌─────────────┐     POST /chat      ┌────────────────────────────────────────────┐
│  Next.js    │ ──────────────────► │  FastAPI  app/main.py  →  api/routes.py    │
│  frontend   │ ◄── citations/timing│  registry/search.py (hybrid retrieval)     │
└─────────────┘                     │    dense (Nomic 768-d) + sparse FTS + RRF  │
                                    │    → backend/app/gateway/ (LLM routing)    │
                                    │    → grounded answer or evidence-only      │
                                    └──────────┬─────────────────────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
            safety_registry.db          Redis + Celery              storage/UNECE/
            (SQLite local / PG docker)  (async ingest)              validated PDFs
                    │
                    └── parser/ + vectorization/structure_chunker.py
                        (clause/table-aware chunks; OCR section promotion)
```

**Gateway chain (serving):** `groq (70B)` → `groq_fast (20B)` → `openrouter_llama` → `openrouter_claude` → **evidence-only** (retrieved passages only, no synthesis).

**PRDs:** `Passive_Safety_Assistant_PRD.md` (chat/RAG) · `Regulation_KnowledgeBase_Registry_PRD.md` (pipeline)

## Quick start (local demo)

### 1. Environment

Copy `.env` and set:

```bash
GROQ_API_KEY=gsk_...                  # primary serving LLM
OPENROUTER_API_KEY=sk-or-...          # failover when Groq rate-limited
ANTHROPIC_API_KEY=sk-ant-...          # authoritative RAGAS judge (when egress allows)
ENABLE_GATEWAY=true
STRUCTURE_CHUNKING=true
USE_MOCK_EMBEDDINGS=false             # Windows: required for real Nomic retrieval
EMBED_BACKEND=transformers
```

### 2. Backend + worker

```powershell
cd H:\AutoSafety_RAG
docker compose up redis -d

# Terminal 1 — API
$env:STRUCTURE_CHUNKING='true'
$env:ENABLE_GATEWAY='true'
$env:USE_MOCK_EMBEDDINGS='false'
$env:EMBED_BACKEND='transformers'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Celery
celery -A scheduler.celery_app worker --loglevel=info --pool=solo
```

Health: `GET http://127.0.0.1:8000/api/v1/health` → `celery_worker: up`  
Ready gate: `GET http://127.0.0.1:8000/api/v1/ready` → `ready: true`

### 3. Frontend

```powershell
cd frontend
# .env.local: NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api/v1
npm run dev
```

Open **http://localhost:3000** — chat enables once `/ready` returns true.

### 4. Docker full stack (optional)

```powershell
docker compose up --build
# API on http://localhost:8002/api/v1
```

First build downloads PyTorch/CUDA (~15 min). Local SQLite + uvicorn is faster for development.

## Ask a question

**UI:** type in the chat box; citations appear in the right sidebar with model + latency.

**API:**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What does UN R14 Annex 6 specify for ISOFIX anchorage points?","top_k":8}'
```

## Corpus & coverage

Live: `GET /api/v1/coverage`

| Metric | Value (2026-06-29) |
|--------|---------------------|
| Chunks indexed | ~14,200 |
| Embedding | `nomic-ai/nomic-embed-text-v1.5` @ **768-d** |
| UNECE regs ingested | **22 / 22** expected |
| Coverage rate | **100%** (all expected regs present) |
| Completeness rate | **86.4%** (19 complete base texts, 3 partial) |

**Partial (series/amendment PDF only — no standalone base text yet):** `UN_R14`, `UN_R29`, `UN_R80`. These are ingested and searchable but flagged partial until a non-series base PDF is added.

## Add a new regulation (supported acquisition path)

Automated UNECE crawling is **not** the primary path on this network (Cloudflare / 403 blocks headless fetch). Use manual acquisition:

1. **Download** the official PDF in a browser from the [UNECE WP.29 register](https://unece.org/transport/vehicle-regulations-wp29/standards) (open network).
2. **Stage** the file under `data/staging/` (must be a valid `%PDF-` per `registry/validation.py`).
3. **Ingest:**
   ```powershell
   python scripts/ingest_offline_reg.py data/staging/UN_Rxx.pdf --regulation-code UN_Rxx --amendment "0N Series"
   ```
4. **Verify:** `GET /api/v1/coverage` and a targeted `/chat` query.
5. **Snapshot** before bulk changes: copy `safety_registry.db` → `data/backups/`.

For OCR-scanned PDFs, use `scripts/ocr_quarantine_ingest.py` per `output/HUMAN_ACTION_RUNBOOK.md`.

## LLM gateway

```
groq (70B) → groq_fast (20B) → openrouter_llama → openrouter_claude → evidence-only
```

- Connect timeout: **3s** (`GATEWAY_CONNECT_TIMEOUT`)
- Circuit breaker after repeated connection failures
- OpenRouter failover logs **+5.5% fee** line in server logs
- Override models: `GROQ_MODEL`, `GROQ_MODEL_FAST`, `OPENROUTER_MODEL_LLAMA`, `OPENROUTER_MODEL_CLAUDE`

## Tests & evaluation

```powershell
python -m pytest tests/ -q
python scripts/run_e2e_demo.py               # live /chat questions → output/e2e_demo_results.md
python scripts/print_query_timing.py "..."   # per-step latency breakdown
```

### RAGAS eval (10 questions, 4 metrics)

**Current judge status:** scores in `output/ragas_evaluation_final.csv` use an **INTERIM Groq judge** (`gpt-oss-20b` fallback after TPD limits on `llama-3.3-70b-versatile`). Direct Anthropic egress is blocked on this network; OpenRouter returned 401 during the interim run. Treat interim precision/recall numbers as directional only.

**Authoritative re-score** (when Anthropic or OpenRouter Claude access works):

```powershell
# Full pipeline (retrieval + judge + PNG) — preferred
python scripts/run_ragas_evaluation_final.py --require-anthropic

# Score-only against existing retrieval snapshot
python scripts/run_ragas_score.py `
  output/ragas_evaluation_final.retrieval.json `
  output/ragas_evaluation_final.csv `
  --require-anthropic `
  --resume
```

See also `output/RUN_AUTHORITATIVE_ELSEWHERE.md` and `output/HUMAN_ACTION_RUNBOOK.md`.

**Retrieval regression check (deterministic, no LLM):**

```powershell
python scripts/diff_retrieval_chunks.py
# → output/retrieval_chunk_diff_report.json
```

Task 2 (2026-06-29): deterministic before/after chunk diff confirmed a **real retrieval regression** (mean Jaccard 0.31), not judge noise. Fixes: scope-chunk deprioritization, cited-section promotion, `source_type=UNECE` only on multi-reg queries. Evidence: `output/retrieval_chunk_diff_report.txt`.

## Known limitations

- **RAGAS judge:** INTERIM Groq only until `ANTHROPIC_API_KEY` or working `OPENROUTER_API_KEY` + Claude model path is available.
- **Rate limits:** multi-reg / multi-market comparison queries (e.g. R44 vs R129, FMVSS vs R94) degrade under Groq TPD; serving may fall back to 20B or evidence-only.
- **OCR section tagging:** R16 had clause-on-own-line mis-tags (§7.6.2 buried in annex blocks) — **fixed** via `parser/structure_extract.py` + R16 re-chunk. **UN_R21, UN_R32, UN_R33** audited dry-run — no silent gap detected.
- **Partial regs:** `UN_R14`, `UN_R29`, `UN_R80` lack standalone base PDFs in corpus (series/amendment only).

## Key env vars

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Primary serving LLM |
| `OPENROUTER_API_KEY` | Failover serving + Claude via OpenRouter |
| `ANTHROPIC_API_KEY` | Authoritative RAGAS judge (direct API) |
| `ENABLE_GATEWAY` | Multi-tier routing (default true) |
| `STRUCTURE_CHUNKING` | Clause/table-aware chunks |
| `USE_MOCK_EMBEDDINGS` | Set `false` on Windows for real Nomic |
| `EMBED_BACKEND` | `transformers` on Windows |
| `NEXT_PUBLIC_API_URL` | Frontend → backend URL |
| `REGISTRY_CORS_ORIGINS` | CORS allowlist (includes localhost:3000) |

## Output artifacts

| File | Role |
|------|------|
| `output/ragas_evaluation_final.csv` | Latest 10-case RAGAS scores (interim judge) |
| `output/ragas_evaluation_final.retrieval.json` | Retrieval snapshot for re-score |
| `output/retrieval_chunk_diff_report.json` | Task 2 deterministic chunk diff |
| `output/CLEANUP_AUDIT.md` | Repo cleanup audit (Phase 2) |
| `output/HUMAN_ACTION_RUNBOOK.md` | Manual acquisition + OCR runbook |
| `output/e2e_demo_results.md` | Live demo capture |
