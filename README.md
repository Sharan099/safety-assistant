# PSA AI — Passive Safety RAG (v2.1)

Production RAG stack for passive safety regulations with an **advanced multi-stage retriever** (query expansion → multi-query → hybrid semantic+BM25+RRF → metadata filtering → parent-child → BGE rerank), **LangGraph** orchestration, **Guardrails AI**, **LangSmith** tracing, and **Grafana/Prometheus** monitoring. Scanned PDFs are ingested with **PaddleOCR / PP-OCR** and **hierarchical chunking**.

> **v2.1 — Grounding & citations** (see [Grounding & citations](#grounding--citations-v21)):
> every answer now carries structured source citations (document, page/section,
> revision), abstains with *"I don't know"* when retrieval confidence is below a
> threshold, separates **legal regulations** from **rating protocols**, and flags
> regulations that have multiple revisions. This is the foundation for the wider
> roadmap (two-tier knowledge base, comparison tables, audit trail, SSO/RBAC).

## Architecture

```
React/Next.js Frontend
        ↓
   API Gateway (nginx :8080)
        ↓
   FastAPI Backend (:8000)
        ↓
   LangGraph Workflow
   ├── Guardrails (input)
   ├── Advanced Retriever
   │     User Query
   │       ↓ Query Expansion        (domain synonyms + intent detection)
   │       ↓ Multi-Query Generation (N query variants)
   │       ↓ Hybrid Retrieval       (Dense + BM25 + RRF, per variant + multi-query fusion)
   │       ↓ Metadata Filtering     (boost chunks matching query intent flags)
   │       ↓ Parent-Child Retrieval (precise child chunks + parent section context)
   ├── Cross-Encoder Reranker        (BAAI/bge-reranker-base)
   ├── Prompt builder
   ├── Groq LLM
   └── Guardrails (output: PII / unsafe warnings)
        ↓
   Response + observability
```

| Layer | Technology |
|--------|------------|
| Frontend | Next.js 14 |
| API Gateway | nginx |
| Backend | FastAPI |
| Orchestration | LangGraph |
| Query expansion | domain synonyms + intent detection (`query_expansion.py`) |
| Multi-query | rule-based variants (optional Groq paraphrases) |
| Embeddings (semantic) | `BAAI/bge-base-en-v1.5` (768-dim) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) + multi-query RRF |
| Metadata filtering | chunk feature flags (`has_loads`, `has_test_procedure`, …) |
| Parent-child | child paragraph match + parent section context |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| LLM | Groq `llama-3.3-70b-versatile` |
| OCR ingestion | PaddleOCR / PP-OCR (RapidOCR ONNX fallback) |
| Observability | LangSmith (query, docs, prompt, response, latency) |
| Monitoring | Prometheus + Grafana (cost, latency, tokens, errors) |
| Evaluation | RAGAS + `tests/test_ragas_evaluation.py` |

> Note: `backend/app/graph/workflow.py` is the **LangGraph** orchestration graph,
> not GraphRAG. GraphRAG (Neo4j KG) was removed.

### Retrieval tuning (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_QUERY_EXPANSION` | `true` | Add domain synonyms (daN, traction, test load, …) |
| `ENABLE_MULTI_QUERY` | `true` | Run 3–4 query variants, fuse with RRF |
| `MULTI_QUERY_COUNT` | `3` | Max extra variants |
| `ENABLE_METADATA_FILTER` | `true` | Boost chunks matching intent flags |
| `METADATA_BOOST` | `0.5` | Score multiplier per matching flag |
| `ENABLE_PARENT_CHILD` | `true` | Attach parent section context to child hits |
| `ENABLE_LLM_MULTI_QUERY` | `false` | Groq paraphrases (uses API tokens) |

## Grounding & citations (v2.1)

The highest-priority requirement: **no source → no claim.** This release adds a
grounding layer on top of retrieval.

**What it does**

- **Structured citations.** Every retrieved passage becomes a citation carrying
  `document`, `page`/`section` (clause number when available, e.g. `§6.4.2`),
  `revision/amendment`, and `doc_type`. Markers `[S1] [S2] …` are injected into
  the LLM context and the prompt requires an inline citation after every claim.
- **Confidence gate / abstention.** If the top retrieval confidence is below a
  threshold, the bot replies *"I don't know — not found in the corpus"* instead
  of generating. Confidence uses the raw semantic cosine (and the cross-encoder
  probability when the reranker is on).
- **Legal vs rating separation.** A document registry tags each source as a
  **legal regulation** (binding — UN/ECE, FMVSS), a **rating protocol**
  (Euro NCAP — not legally binding), or an engineering reference. Context is
  grouped by type and the prompt forbids blurring them. The UI shows a
  `Legal` / `Rating` badge per source.
- **Multi-revision flag.** When a cited legal regulation has multiple revisions
  (e.g. UN R14 Rev.7, UN R16 Rev.10), the answer carries a flag prompting the
  user to confirm the applicable version.

**Where it lives**

| File | Responsibility |
|------|----------------|
| `backend/app/core/document_registry.py` | Source of truth: doc type, authority, indexed revision, known revisions |
| `backend/app/retrieval/citations.py` | Page/section parsing, citation building, grounding assessment, answer flags |
| `backend/app/graph/workflow.py` | Grounded context, grounding gate, citation-aware prompt |
| `backend/app/api/routes.py` | Returns `citations`, `flags`, `grounding` |
| `frontend/src/app/page.tsx` | Citation cards, Legal/Rating badges, revision/abstain banners, export |

**Config (`.env`)**

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_GROUNDING_GATE` | `true` | Abstain when retrieval confidence is low |
| `GROUNDING_MIN_SEMANTIC` | `0.45` | Min top semantic cosine to answer |
| `GROUNDING_MIN_RERANK_PROB` | `0.5` | Min rerank probability (when reranker on) |
| `REQUIRE_CITATIONS` | `true` | Require inline `[S#]` citations in answers |

**Verify**

```powershell
# Retrieval + citation + grounding, no LLM tokens needed:
conda run -n rag python -c "from backend.app.retrieval.hybrid import HybridRetriever; from backend.app.retrieval.citations import build_citations, assess_grounding; r=HybridRetriever(); res=r.retrieve('UN R14 seat belt anchorage strength'); print(assess_grounding(res['documents'], reranker_used=False, min_semantic=0.45, min_rerank_prob=0.5)); print(build_citations(res['documents'])[0]['label'])"
```

Expected: a grounded query returns `should_abstain: False` with a citation like
`UN R14 (Revision 7 (09 series of amendments)), §6.4.2`; an out-of-scope query
(e.g. *"best pizza topping"*) returns `should_abstain: True`.

## Quick start

### 1. Environment

```bash
cp .env.example .env
# Set GROQ_API_KEY (required for LLM responses)
# Optional: LANGSMITH_API_KEY, LANGSMITH_TRACING=true
```

### 2. Docker (full stack)

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| App (gateway) | http://localhost:8080 |
| API docs | http://localhost:8080/docs |
| Grafana | http://localhost:3001 (admin/admin) |
| Prometheus | http://localhost:9090 |

### 3. Local development

ML deps (torch, sentence-transformers, paddle/rapidocr) live in a conda env named
**`rag`**. Run all Python from that env so the models resolve correctly.

```bash
conda activate rag
pip install -r requirements.txt

# Backend (run inside the rag env)
conda run -n rag uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

Frontend: http://localhost:3000

The frontend defaults to:

- `http://localhost:8000/api/v1` when running locally on port `3000`
- `/api/v1` when served through the gateway on port `8080`

You can override this with `NEXT_PUBLIC_API_URL`.

### Troubleshooting

**`ModuleNotFoundError: sentence_transformers` / `No module named 'rapidocr'`**
You are running anaconda *base* instead of the `rag` env. Use `conda run -n rag …`.

**Crash with exit code `0xC0000005` (Windows access violation)**
This is a torch/OpenMP DLL clash. The repo `.env` sets the fix:
```env
KMP_DUPLICATE_LIB_OK=TRUE
OMP_NUM_THREADS=1
```
For OCR, also use `OCR_BACKEND=rapidocr` (PP-OCR via ONNX) instead of native Paddle.

**Slow / empty chat**
1. Set **`GROQ_API_KEY`** in `.env` (required for answers).
2. If the embedding model still hangs, fall back to BM25-only:
   ```env
   DISABLE_SEMANTIC=true
   ENABLE_RERANKER=false
   ```
3. Restart backend; check `http://localhost:8000/api/v1/health` → `groq_configured: true`.

> Note: `BAAI/bge-reranker-base` runs on CPU (~10–30 s per query). For low-latency
> local chat, set `ENABLE_RERANKER=false` — the hybrid + RRF ranking is still used.

### 4. Verify retrieval & run evaluation

**20 questions (recommended, with Groq LLM):**

```powershell
.\scripts\run_evaluation_20.ps1
```

**70 questions (full benchmark):**

```bash
conda run -n rag python tests/generate_test_cases_70.py
conda run -n rag python tests/run_full_evaluation.py
# or: .\scripts\run_evaluation.ps1
```

If Groq daily token limit is hit, use retrieval-only proxies:

```powershell
$env:EVAL_SKIP_LLM='true'
conda run -n rag python tests/run_full_evaluation.py
```

Outputs under `output/evaluation/`:

| File | Description |
|------|-------------|
| `rag_eval_20_results.json` | **20-Q run** (Groq answers) |
| `eval_*_20.png` | Charts for 20-Q run |
| `rag_full_evaluation_results.json` | 70-Q run |
| `eval_ragas_metrics.png` | 70-Q charts (no `_20` suffix) |

## Project layout

```
AutoSafety_RAG/
├── backend/app/          # FastAPI + LangGraph + hybrid retrieval + reranker + guardrails
│   ├── retrieval/        #   hybrid.py (BM25+semantic+RRF), reranker.py (BGE)
│   ├── graph/            #   LangGraph workflow
│   ├── guardrails/       #   input/output validation
│   ├── core/             #   settings, services, observability (LangSmith)
│   └── metrics/          #   Prometheus metrics
├── frontend/             # Next.js UI
├── gateway/              # nginx API gateway
├── monitoring/           # Prometheus config + Grafana provisioning/dashboard
├── data/                 # PDFs + ingestion pipeline (paddle_ocr_converter, hierarchical_chunker, embed_chunks)
├── scripts/              # run_ingestion_pipeline.py, verify_retrieval.py
├── output/               # Markdown, chunks, embeddings, evaluation artifacts
├── config.py             # Shared model & path config
└── tests/                # RAGAS evaluation + test cases
```

## Offline data pipeline (PaddleOCR + hierarchical chunking)

Recommended pipeline for **scanned PDFs** (low memory on Windows):

```bash
pip install -r requirements.txt

# Full pipeline: PDF -> Markdown (PaddleOCR) -> hierarchical chunks -> embeddings
python scripts/run_ingestion_pipeline.py

# Faster dev run (UN R14 + R16 only):
python scripts/run_ingestion_pipeline.py --only UN_R14 UN_R16

# Reuse existing markdown, only re-chunk + embed:
python scripts/run_ingestion_pipeline.py --skip-docling
```

**PaddleOCR / PP-OCR** (default `OCR_ENGINE=paddle`):

| Technique | Setting | Purpose |
|-----------|---------|---------|
| Low-DPI page cache | `OCR_DPI=150` | Smaller PNGs in `output/page_cache/` |
| Small OCR batches | `OCR_BATCH_PAGES=4` | Process 4 pages, then `gc` — avoids OOM |
| Skip text pages | `OCR_SKIP_TEXT_PAGES=true` | Use embedded text when present |
| Windows fallback | `OCR_BACKEND=rapidocr` | PP-OCR via ONNX if native Paddle crashes |

`OCR_BACKEND`: `auto` | `paddle` | `rapidocr` (use `rapidocr` on Windows if you see exit code `0xC0000005`).

Alternative: `OCR_ENGINE=docling` or `OCR_ENGINE=pymupdf`.

Steps:

| Step | Script | Output |
|------|--------|--------|
| 1. OCR → Markdown | `data/docling_converter.py` → `data/paddle_ocr_converter.py` | `output/markdown/*.md` |
| 2. Hierarchical chunk | `data/hierarchical_chunker.py` | `output/regulation_chunks.json` |
| 3. Embed | `data/embed_chunks.py` | `output/regulation_embeddings.json` |

After ingestion, **restart the backend** to load new artifacts.

Set `EMBEDDING_BATCH=4` in `.env` if embedding still runs out of memory on CPU.

## Evaluation results

### Latest: 20-question run (RAGAS, Groq-judged)

`output/evaluation/rag_eval_20_results.json` — **15 regulation + 5 guardrail**.
RAGAS metrics below are **LLM-judged by Groq** (all 60 judge jobs completed).

| Metric | Score |
|--------|-------|
| **Faithfulness** | **0.800** |
| **Answer relevancy** | **0.664** |
| **Context precision** | **0.674** |
| **Context recall** | **0.733** |
| **Overall score** (mean of four) | **0.718** |

**Ablation:** hybrid context recall **+32.3%** vs semantic-only (0.357 → 0.472).  
**Guardrails:** 100% injection blocked; 100% out-of-scope safe; **0%** hallucination proxy (hybrid).  
**Latency:** retrieval p95 **8.8 s**, pipeline p95 **12.4 s**; ~872 prompt + 227 completion tokens/query; **$0.00069**/query.

> Note: during answer generation the Groq free-tier rate limit was reached after
> the first few questions, so the remaining answers fell back to retrieval proxies.
> The RAGAS scoring above still ran fully on Groq. Re-run when the quota resets for
> all-LLM answers: `conda run -n rag python tests/run_full_evaluation.py`.

#### 20-question charts

![RAGAS metrics (20Q)](output/evaluation/eval_ragas_metrics_20.png)

![Ablation (20Q)](output/evaluation/eval_ablation_comparison_20.png)

![Overall scorecard (20Q)](output/evaluation/eval_overall_scorecard_20.png)

![Guardrails (20Q)](output/evaluation/eval_guardrails_20.png)

![Latency distribution (20Q)](output/evaluation/eval_latency_distribution_20.png)

### Prior: 70-question run (retrieval proxy, no Groq quota)

`output/evaluation/rag_full_evaluation_results.json` — **60 regulation + 10 guardrail**  
Mode: `EVAL_SKIP_LLM=true` (proxy answers). Overall score **0.401**; hybrid recall **+22%**.

### Corpus scale (indexed today)

| Stat | Value |
|------|-------|
| Regulation PDFs in repo | **37** |
| PDFs indexed (OCR markdown) | **2** (UN R14, UN R16) |
| Pages in markdown | **148** |
| Chunks indexed | **1,572** |
| Embedding vectors | **1,572** (`BAAI/bge-base-en-v1.5`) |

### RAGAS-style metrics (hybrid + RRF + BGE rerank)

| Metric | Score |
|--------|-------|
| **Faithfulness** | **0.823** |
| **Answer relevancy** | **0.177** |
| **Context precision** | **0.207** |
| **Context recall** | **0.397** |
| **Overall score** (mean of four) | **0.401** |

### Ablation: hybrid beats semantic-only

| Retrieval | Context recall | Context precision | Avg latency |
|-----------|----------------|-------------------|-------------|
| Semantic only | 0.326 | 0.160 | 632 ms |
| **Hybrid + RRF + rerank** | **0.397** | **0.207** | 10.1 s |

**Hybrid search improved context recall by +22.0%** over semantic-only retrieval by adding BM25 + RRF + cross-encoder reranking.

### Guardrails (measured)

| Effect | Result |
|--------|--------|
| Injection/jailbreak input blocked | **100%** (6/6) |
| Out-of-scope safe handling | **75%** (3/4) |
| Hallucination proxy (semantic → hybrid) | **5.0% → 1.7%** on regulation set |

### Latency / cost (observability-style)

| Metric | Value |
|--------|-------|
| Retrieval **p95** | **333 ms** |
| Full pipeline **p95** (incl. BGE rerank) | **19.3 s** |
| Est. cost / query (Groq proxy rates) | **$0.00045** |

### Charts

![RAGAS metrics](output/evaluation/eval_ragas_metrics.png)

![Ablation](output/evaluation/eval_ablation_comparison.png)

![Overall scorecard](output/evaluation/eval_overall_scorecard.png)

Re-run with live Groq answers for LLM-judged RAGAS: clear `EVAL_SKIP_LLM`, ensure API quota, then `conda run -n rag python tests/run_full_evaluation.py`.

## Observability & monitoring

The system emits two complementary signal streams:

### 1. Tracing — LangSmith (per-request detail)

`backend/app/core/observability.py` wraps the pipeline so each request records
**query → retrieved docs → prompt → response → latency** as a LangSmith run.

Enable it in `.env`:
```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=autosafety-rag
```
Traces appear at [smith.langchain.com](https://smith.langchain.com) under the project.
When disabled the wrappers are no-ops (zero overhead).

### 2. Metrics — Prometheus + Grafana (aggregate health)

The backend exposes Prometheus metrics at **`GET /metrics`** (defined in
`backend/app/metrics/prometheus.py`):

| Metric | Type | Meaning |
|--------|------|---------|
| `rag_request_duration_seconds` | histogram | end-to-end latency by endpoint/status |
| `rag_retrieval_duration_seconds` | histogram | hybrid retrieval latency |
| `rag_llm_duration_seconds` | histogram | Groq generation latency |
| `rag_tokens_prompt_total` / `rag_tokens_completion_total` | counter | token usage |
| `rag_estimated_cost_usd_total` | counter | cost proxy (Groq llama-3.3-70b rates) |
| `rag_errors_total{error_type}` | counter | error rate |
| `rag_guardrail_blocks_total{reason}` | counter | blocked prompts (injection/jailbreak) |
| `rag_active_requests` | gauge | in-flight requests |

**Flow:** `FastAPI /metrics` → Prometheus scrapes every 15 s
(`monitoring/prometheus.yml`) → Grafana auto-provisions the Prometheus datasource
and a dashboard (`monitoring/grafana/provisioning/`).

### How to visualize

```bash
docker compose up --build
```

| Tool | URL | Use |
|------|-----|-----|
| Grafana | http://localhost:3001 (admin/admin) | dashboards: latency, cost, tokens, errors |
| Prometheus | http://localhost:9090 | raw metric queries / ad-hoc PromQL |
| Raw metrics | http://localhost:8000/metrics | scrape endpoint |

The **"AutoSafety RAG"** dashboard loads automatically in Grafana. Useful PromQL:

```promql
# p95 end-to-end latency
histogram_quantile(0.95, sum(rate(rag_request_duration_seconds_bucket[5m])) by (le))

# estimated $/hour
rate(rag_estimated_cost_usd_total[1h]) * 3600

# error rate
sum(rate(rag_errors_total[5m]))
```

### Future visualization ideas

- **Alerting:** Grafana alert rules on p95 latency, error rate, or hourly cost
  (e.g. notify Slack when `rate(rag_errors_total[5m]) > 0`).
- **Retrieval-quality panels:** export per-query `semantic_count` / `bm25_count` /
  `rerank_score` as metrics to chart hybrid contribution and rerank lift over time.
- **RAGAS over time:** push `tests/test_ragas_evaluation.py` results into Prometheus
  (pushgateway) or a CI job to trend context recall / faithfulness per build.
- **Tracing dashboards:** connect LangSmith datasets to Grafana, or add OpenTelemetry
  spans for distributed traces across gateway → backend → Groq.

## Guardrails

- Blocks prompt injection & jailbreak patterns on input
- Warns on possible PII or unsafe content in responses

## Screenshots

Live screenshots of the running system (chat UI, guardrails, and observability).
All images live in `output/screenshots/`.

### Chat interface (PSA AI — Next.js)

UN R14 regulation answer with grounded sources and per-stage latency:

![Chat — What is UN R14](output/screenshots/PSA_1.png)

UN R16 specification answer:

![Chat — What does UN R16 specify](output/screenshots/PSA_2.png)

Seat belt anchorage requirements (parent-child + reranked context):

![Chat — Seat belt anchorage requirements](output/screenshots/PSA_3.png)

### Guardrails in action

Prompt-injection attempt ("Ignore all previous instructions…") blocked at input:

![Guardrail — prompt injection blocked](output/screenshots/Test_prompt_injection.png)

Out-of-scope question ("Who won the FIFA World Cup 2022?") correctly answered
with *"Information not found in regulations"* — grounding / scope boundary:

![Scope boundary — out-of-scope question](output/screenshots/Text_boundry.png)

### Observability — Grafana & Prometheus

Operations dashboard (request/LLM latency, cost, token usage, error rate, active requests):

![Grafana dashboard](output/screenshots/Dashboard_1.png)

Guardrail blocks spiking on the dashboard during prompt-injection tests:

![Grafana — guardrail blocks](output/screenshots/Prompt_Injection.png)

Prometheus scrape target (`autosafety-rag-backend`) healthy / UP:

![Prometheus targets](output/screenshots/Prometheus.png)

## License

Internal / research use — passive safety engineering assistant.
