# Passive Safety RAG

Structure-aware RAG over UNECE passive-safety regulations
(**Docling** parse → **Qdrant** hybrid retrieval → grounded generation).

Indexed corpus today: **UN-ECE-R94**, **R95**, **R16**, **R129**.

## Layout

```
ingestion/       # PDF → Docling → validate/OCR → chunk → enrich → embed/upsert
retrieval/       # intent router + hybrid RRF + rewrite/rerank/expand + biases
generation/      # LLM client + grounded answers + deterministic compliance
api/             # FastAPI: /chat SSE, /agent, /pdf, /citation, upload, metrics
app/             # CLI (`python -m app.ask` / passive-safety-ask)
frontend/        # Next.js chat + PDF highlight + metrics rail
agent/           # citation-strict multi-step tools (compare / report / …)
eval/            # golden_set.jsonl + run_full / run_retrieval_only + dashboards
observability/   # per-query traces + cost
config/          # Portkey fallbacks, prices, acronyms, domain maps
data/            # PDFs, Docling JSON, local Qdrant, traces, limits
scripts/         # rebuild helpers, finalize-from-partial, diagnostics
tests/
```

CLI entrypoints (also via `uv run`): `passive-safety-ingest`, `passive-safety-ask`,
`passive-safety-eval`, `passive-safety-api`, `passive-safety-agent`.

## Setup

```bash
uv sync
cp .env.example .env
```

Place regulation PDFs under `data/pdfs/` (e.g. `UN_R94.pdf`, `UN_R95.pdf`,
`UN_R16.pdf`, `UN_R129.pdf`).

## Quick start

```bash
# 1) Ingest one regulation (idempotent by regulation_id)
python -m ingestion.run \
  --pdf data/pdfs/UN_R94.pdf \
  --regulation-id "UN-ECE-R94" \
  --revision "Rev.3"

# Or rebuild the full indexed corpus from existing Docling exports:
python -m ingestion.stage4_rebuild

# 2) Ask (default LLM_PROVIDER=mock — no provider quota used)
python -m app.ask "What is the HIC15 limit in R94?"
python -m app.ask "What is the HIC15 limit in R94?" --json
```

## Pipeline

| Step | Module | Behaviour |
|------|--------|-----------|
| 1 | `ingestion.parse` | Docling PDF → `DoclingDocument` (+ JSON under `data/docling/`). |
| 1b | `ingestion.extract` + validator | Structured extract + flags; pages needing LightOnOCR. |
| 1c | `ingestion.vlm_figure_pass` | LightOnOCR-2 on flagged / opt-in pages; unresolved → `data/ocr_review_queue/`. |
| 2 | `ingestion.chunk` | One chunk per clause/sub-clause; tables as atomic markdown; parent–child links. |
| 2b | `ingestion.describe_figures` | Optional `content_type=figure` chunks (Portkey `figure_describe.json`). |
| 3 | `ingestion.enrich` | Prepends `From {id} {rev} §{num} {title}:` (+ role hints) before embedding. |
| 4 | `ingestion.embed_upsert` | `bge-small-en-v1.5` dense + BM25 sparse → Qdrant `regulations`. |
| 5 | `retrieval.router` | Classify intent (factual / compliance / checklist / design / scope / …). |
| 6 | `retrieval.retrieve` | Hybrid RRF → optional rewrite/rerank/small-to-big → intent budgets + biases. |
| 7 | `generation.answer` | Grounded JSON segments + citations; deterministic compliance PASS/FAIL when applicable. |
| 8 | `app.ask` / `api` / `frontend` | CLI, SSE chat, or Next.js UI with PDF highlight. |

Corpus rebuild report: `data/stage4_rebuild_report.json`.
“Stage 5” in this repo means **retrieval evaluation** (`eval.run_retrieval_only` /
`tests/test_stage5_retrieval.py`), not a second rebuild CLI.

## Retrieval

`retrieve()` is intent-routed (`retrieval/router.py`: regex fast-path, optional
`QUERY_ROUTER_LLM`). Shared stages:

1. **Rewrite** (`retrieval/rewrite.py`) — acronym expand + subquery split.
   Heuristic rewrite on by default (`RETRIEVAL_REWRITE=1`); LLM split off by default
   (`RETRIEVAL_REWRITE_LLM=0`).
2. **Hybrid RRF** — dense + Qdrant BM25 → ~`HYBRID_TOP_K=30` (weighted sparse boost
   for exact regulatory terms).
3. **Rerank** (`retrieval/rerank.py`) — local BGE reranker (or Cohere) → top ~5
   (enumerative uses a wider window).
4. **Small-to-big** (`retrieval/expand.py`) — expand leaves to parent sections
   (skipped on some compliance paths).
5. **Context budget** — default 5 chunks / 3k tokens; intent overrides (checklist /
   enumerative / design are wider).

Specialized biases (do not demote definitions on definition-seeking asks):

| Path | Module | When |
|------|--------|------|
| Value / limit prefer | `retrieval/value_limit.py` | Pass/fail or measured-value vs limit (not “What does X mean?”) |
| Enumerative topic bias | `retrieval/enumerative.py` | “List every …”, door/requirement surveys |
| Multi-regulation | `retrieval/multi_regulation.py` | Plural / “in general” surveys across indexed regs |
| Multi-criterion | `retrieval/multi_criterion.py` | Several named injury criteria in one question |

Re-ingest after collection schema changes (named vectors `dense` + `bm25`):

```bash
python -m ingestion.run --pdf data/pdfs/UN_R94.pdf --regulation-id "UN-ECE-R94" --revision "Rev.3"
# or: python -m ingestion.stage4_rebuild
```

## Generation & LLM client

`generation/answer.py` routes by intent. Compliance checks use
`generation/compliance.py` for **deterministic** measured-value vs
`data/limits/*.json` PASS/FAIL; the LLM only phrases around those verdicts.
Failure modes include `retrieval_miss`, `grounding_rejected`, and
`numeric_hallucination`.

`generation/llm_client.py` — OpenAI-compatible client pointed at the local Portkey gateway:

- `LLM_PROVIDER=mock|groq` (default **mock** in `.env.example`)
- Per-task Portkey Config JSON under `config/portkey/`:
  - `query_rewrite.json` — Groq 8B → NIM 8B → Gemini Flash (+ simple cache, 24h TTL)
  - `final_answer.json` — Groq 70B → Nemotron Super 49B → Gemini Flash → OpenRouter free (+ cache, retry)
  - `eval_judge_pinned.json` — **eval scoring only**: single `gemini-2.5-flash`, no production fallback chain
  - `figure_describe.json` — figure VLM describe
- Cache invalidation: live requests send `x-portkey-cache-namespace` including ingest
  `cache_version` (bumped on successful PDF upload)
- Optional local SQLite answer cache (`ANSWER_CACHE=1`) is off by default — Portkey is primary
- Mock mode still uses a small disk cache under `data/llm_cache/` for deterministic local runs

**Eval vs production:** FreeLLMAPI (`localhost:3001`) is optional overflow for eval
infra only. Generation / retrieval / app never route through it; do not reuse eval
judge wiring for production traffic.

## Local Portkey AI Gateway (LLM proxy)

The stack routes live LLM traffic through the [open-source Portkey AI Gateway](https://github.com/portkey-ai/gateway)
on **localhost**. Image: `portkeyai/gateway`. Basic proxying, fallbacks, load-balancing,
and local caching do **not** require a Portkey cloud account.

Regulation content leaves your machine **only** to whichever LLM provider serves the
request. The local gateway does not send bodies to Portkey cloud.

```bash
docker run --rm -p 8787:8787 --name portkey-gateway portkeyai/gateway:latest
# Or: docker compose up -d portkey
```

| Endpoint | URL |
|----------|-----|
| OpenAI-compatible API | `http://localhost:8787/v1` |
| Local console | `http://localhost:8787/public/` |

```bash
# .env — host processes (uvicorn, CLI, eval)
PORTKEY_GATEWAY_URL=http://localhost:8787/v1
LLM_PROVIDER=groq
GROQ_API_KEY=...
# NVIDIA_API_KEY=...   # optional fallbacks
# GOOGLE_API_KEY=...
# OPENROUTER_API_KEY=...
```

`docker compose` sets `PORTKEY_GATEWAY_URL=http://portkey:8787/v1` for the `api`
service. `LLM_PROVIDER=groq` posts to `{PORTKEY_GATEWAY_URL}/chat/completions` with
`x-portkey-config` set to the task JSON. Targets without an API key are skipped.
`nvidia_nim` maps to Portkey’s OpenAI provider + `https://integrate.api.nvidia.com/v1`.

**NIM / Gemini latency:** Nemotron Super forces `enable_thinking=false` (+ optional
`/no_think`). Gemini 2.5 Flash forces thinking off with a short per-target timeout so
slow legs fall through on HTTP 408. See `scripts/benchmark_gemini_latency.py` and
[`scripts/README.md`](scripts/README.md).

**E2E fallback proof** (invalid Groq key → next provider + cited answer):

```bash
docker compose up -d portkey
uv run python scripts/manual_test_portkey_fallback.py
```

## Agent loop (compare / report)

Citation-strict multi-step agent (`agent/`):

| Tool | Purpose |
|------|---------|
| `retrieve` | Grounded passages with citations |
| `compare_regulations` | Cited comparison table |
| `lookup_table` | Criterion / table lookup |
| `draft_report` | Fully-cited engineering memo |
| `design_implication` / `applicability` / `checklist_gen` / `retest_scope` | Intent-aligned tools |

Flow: **decompose → retrieve per step → reason → synthesize**. Every step runs a
citation gate; uncited factual claims are stripped.

```bash
python -m agent.run "Compare R94 vs R95 chest deflection limits"
python -m agent.run "Gap analysis: our test setup vs R95 requirements" --json
python -m eval.agent_eval

curl -X POST http://127.0.0.1:8000/agent -H "Content-Type: application/json" \
  -d '{"task":"Compare R94 vs FMVSS 208 HIC limits"}'
# also: POST /agent/compare  POST /agent/report
```

Optional LangGraph wrapper: `uv sync --extra agent` then `agent.graph.run_agent_graph`.
UI: toggle **Agent on** in the chat header.

## Observability & cost

Every `/chat` query writes a trace under `data/traces/` (and optionally to Langfuse)
with tokens, model, embedding/rerank counts, latency, and `cost_usd`.

Prices live in [`config/prices.json`](config/prices.json) — update there, never hardcode.

```bash
GET /metrics/{trace_id}
GET /metrics/aggregate
```

UI: ◇ rail → metrics panel.

| Lever | Flag / behaviour |
|-------|------------------|
| Prompt cache | `PROMPT_CACHE=1` — static system prompt reuse |
| Answer cache | Portkey simple cache + `cache_version` namespace; optional `ANSWER_CACHE=1` SQLite |
| Model routing | rewrite=`GROQ_SMALL_MODEL`, answer=`GROQ_LARGE_MODEL` |

## Hardening & Docker

- Input validation on `/chat` (length, control chars, injection heuristics)
- Graceful “not found in regulations” SSE event + UI banner
- Quality gate: `.github/workflows/quality-gate.yml` runs `eval.run_full` on
  `eval/ci_gate.jsonl` against `eval/thresholds.yaml`
- Stack: Qdrant `:6333` + Portkey `:8787` + API `:8000` + frontend `:3000`
  (+ optional FreeLLMAPI on `127.0.0.1:3001` for eval overflow only)

```bash
docker compose up --build
# ingest with QDRANT_URL=http://localhost:6333
# LLM_PROVIDER=groq → local Portkey on :8787
# Gateway-only: docker compose up -d portkey
```

## Chat UI (PDF citation highlight)

```bash
# Terminal A — API
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal B — frontend
cd frontend
npm install
# optional: NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

Open http://127.0.0.1:3000 — ask a question, click a citation chip like
`[R94 §5.2.1, p.12]`, and the right pane opens PDF.js on that page with the
chunk `bounding_box` highlighted.

## Eval

Canonical full suite: **`python -m eval.run_full`** (`passive-safety-eval`).
Per-category scoring uses `eval/scoring/ragas_scorer.py`, `security_scorer.py`, and
`custom_checks.py`, gated by [`eval/thresholds.yaml`](eval/thresholds.yaml).
Metrics are **never blended across categories**.

### Golden set (32 questions)

Canonical set: [`eval/golden_set.jsonl`](eval/golden_set.jsonl) — **32** questions across
**11** categories (30 curated cases + `fig_001` / `fig_002` figure-content probes).

| Category | n | Notes |
|----------|---|--------|
| factual_lookup | 5 | includes figure probes |
| compliance_check | 3 | critical |
| numeric_safety | 3 | critical |
| cross_regulation | 3 | critical |
| guardrail | 3 | critical (ASR) |
| prompt_injection | 3 | critical (ASR) |
| design_implication | 3 | |
| hallucination_probe | 3 | |
| multi_hop | 2 | |
| enumerative | 2 | |
| out_of_scope | 2 | |

Prior 152-case set: [`eval/golden_set_152_archive.jsonl`](eval/golden_set_152_archive.jsonl).
Pre-rebuild run folders: `eval/results_archive_pre_rebuild/`. CI subset:
`eval/ci_gate.jsonl`.

### Retrieval metrics (reporting convention)

Headline MRR / recall@5 / precision@5 / NDCG are computed on **chunk-gold categories
only**:

`factual_lookup`, `compliance_check`, `multi_hop`, `enumerative`, `cross_regulation`,
`design_implication`.

These are **excluded** from retrieval quality aggregates (no single correct-chunk key):

`numeric_safety`, `guardrail`, `prompt_injection`, `hallucination_probe`, `out_of_scope`.

Enumerative primary signal is `chunk_recall@20` (coverage), not MRR.
`eval.run_retrieval_only` and both dashboards label the two numbers separately —
never blend them into one score.

### Latest full-run dashboards

From run `20260807T232542Z` (32 questions; gate status **NOT PRODUCTION READY**):

- Pass/fail readiness: [`eval/results/20260807T232542Z/dashboard.png`](eval/results/20260807T232542Z/dashboard.png)
- RAGAS quality: [`eval/results/20260807T232542Z/ragas_dashboard.png`](eval/results/20260807T232542Z/ragas_dashboard.png)

`eval/results/latest.json` may point at either a full-run pointer or a retrieval-only
scorecard — treat timestamped / `--run-id` artifacts as authoritative.

### Live entry points

| Module | Role |
|--------|------|
| `python -m eval.preflight_check` | Ping each provider in the Portkey fallback chain |
| `python -m eval.smoke_subset` | 5-case full-pipeline smoke |
| `python -m eval.run_full` | Preflight → smoke → batched golden set + `thresholds.yaml` gate |
| `python -m eval.run_full --fill-gaps RUN_ID` | Full pipeline for missing + failed only; resumable |
| `python -m eval.run_full --only-failed RUN_ID` | Re-run failing cases; keep passes |
| `python -m eval.run_full --case-id ID` | Restrict a full run to specific case id(s) |
| `python -m eval.render_dashboard` | PNG pass/fail dashboard (`--run-id` / `--partial`) |
| `python -m eval.render_ragas_dashboard` | PNG RAGAS quality (0–1) |
| `python -m eval.run_retrieval_only` | Zero-LLM retrieval scorecard |
| `python scripts/finalize_from_partial.py --run-id RUN_ID` | Aggregate `partial_results.jsonl` → `results.json` (no LLM) |

`eval.run` was **removed** — use `run_full` for generation gates and
`run_retrieval_only` for retrieval-only scorecards.

Per-case isolation: process worker (`eval/case_timeout.py`), on by default
(`EVAL_CASE_TIMEOUT=1`, `EVAL_CASE_TIMEOUT_S=900`).

```bash
uv sync --extra eval

python -m eval.preflight_check
python -m eval.smoke_subset
python -m eval.run_full --yes
# or: passive-safety-eval --yes

python -m eval.render_dashboard --run-id 20260807T232542Z
python -m eval.render_ragas_dashboard --run-id 20260807T232542Z

python -m eval.run_retrieval_only --tag hybrid --profile hybrid
python scripts/finalize_from_partial.py --run-id <run_id>
```

Results: `eval/results/<run_id>/results.json` (+ `partial_results.jsonl` for resume).

### RAGAS metric caveats

- **`ground_truth_reference_chunks`** supply regulation excerpts used as RAGAS
  `ground_truth` (needed for `context_recall`). Prefer verbatim corpus excerpts.
- Category `ragas_averages` skip NaN per metric and record coverage (`ragas_coverage`).
- Faithfulness NaN skews toward longer multi-claim / structured answers — treat as a
  likely judge/claim-decomposition failure mode, not a short-answer prompt issue.

## Tests

```bash
uv run pytest tests/ -q
```
