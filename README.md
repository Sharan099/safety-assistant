# Passive Safety CAE Investigation Agent

An engineering investigation workstation for passive-safety / occupant-protection
CAE engineers — not a chatbot. It helps answer questions like *"why did chest
deflection increase between Run A and Run B?"* with evidence-backed, traceable
analysis: quality gates, comparability checks, configuration diffs, signal
analysis, divergence detection, citation-grounded regulatory/solver retrieval,
structured LS-DYNA deck queries, and a contextual investigation Copilot — with
an engineer review step before any conclusion is recorded.

## Status

**V1 vertical slice + Copilot + Level 3 research-grade knowledge layer.**
Not a production deployment — see Limitations below.

- **V1 core workflow** (backend + Next.js UI, port 3010): select two runs →
  quality gate → comparability → configuration diff → signal analysis/
  divergence → evidence → agent-drafted hypothesis → engineer review.
- **Investigation Copilot**: a collapsible chat panel inside the investigation
  workspace (not a separate `/chat` page) streaming real-time LangGraph
  workflow visibility over SSE.
- **Level 3**: connects the real public engineering corpus under
  `Knowledge source/` (28 files, 1.3 GB — NHTSA vehicle/dummy/restraint FE
  models, THOR-05F qualification package, OpenRadioss ModelExchange,
  UN regulations, LS-DYNA manuals) to a validated ingestion pipeline,
  a real LS-DYNA parser, mandatory Structured + Hybrid RAG (BM25 + dense +
  RRF + reranker, plus separate structured CAE search), and an evaluated
  retrieval pipeline. See `PRD_LEVEL3.md` / `TRD_LEVEL3.md` /
  `CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md` and the Level-3 section below.

See `IMPLEMENTATION_PLAN.md` for the V1 phased build order and `docs/ADR/`
for every real decision made along the way (13 ADRs).

## Start here

Read these before changing anything — they are the product/technical
contract, not background reading:

1. `PRD.md` / `TRD.md` — V1 product & technical requirements
2. `PRD_LEVEL3.md` / `TRD_LEVEL3.md` — Level-3 requirements
3. `APP_FLOW.md` — the investigation flow and states
4. `BACKEND_SCHEMA.md` — the V1 data model (Level-3 `cae_*` tables:
   `packages/domain/cae.py`)
5. `UI_UX_DESIGN_BRIEF.md` — the UI contract
6. `IMPLEMENTATION_PLAN.md` — V1 phase-by-phase execution order
7. `ENVIRONMENT_SETUP.md` — tooling, OKF architecture
8. `CLAUDE.md` / `CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md` — standing AI-development rules
9. `docs/ADR/` — every real decision, in order

## Repository layout

```
apps/api             FastAPI backend — runs, investigations, knowledge search, copilot
apps/web              Next.js investigation workspace (port 3010)
packages/domain        Core + copilot + CAE entities, Alembic migrations
packages/analysis        Deterministic, LLM-free CAE analysis (no LLM calls)
packages/ingestion         PDF ingestion (PyMuPDF) + archive inspection + QA
packages/cae                 LS-DYNA keyword scan + lexer/parser/include-graph
packages/retrieval              BM25 + pgvector + RRF + reranker + structured CAE search
packages/agent                    LLMProvider + LangGraph investigation agent + Copilot
knowledge/            Canonical corpus registry + OKF concepts (source PDFs gitignored)
Knowledge source/     Immutable original corpus (local only, gitignored — see below)
data/                 Synthetic runs, Parquet signal data, artifacts (gitignored, regenerable)
evals/                Golden datasets + evaluation harnesses
docs/ADR/             Architecture decision records (13)
```

## Getting started

```powershell
uv sync                                          # install Python deps
docker compose up -d postgres                    # port 5433 — see docs/ADR/0004
uv run alembic upgrade head                      # apply the domain + CAE schema
uv run python scripts/generate_synthetic_dataset.py   # SCN-001..010 -> Postgres + Parquet
uv run python scripts/ingest_level3_pdfs.py       # all 16 registered PDFs (bounded, 20p each)
uv run python scripts/index_knowledge.py          # embed ingested chunks (idempotent)
uv run python scripts/ingest_level3_cae_decks.py  # 4 real LS-DYNA deck families -> cae_*
uv run python scripts/generate_okf_concepts.py    # OKF concept files under knowledge/07_okf/
uv run uvicorn apps.api.main:app --port 8010      # backend, port 8010 (docs/ADR/0004)

# in another terminal:
cd apps/web
cp .env.local.example .env.local
npm install
npm run dev                                       # frontend, http://localhost:3010
```

```powershell
uv run pytest              # 187 tests (1 skipped)
uv run ruff check .        # lint
uv run mypy apps packages scripts tests conftest.py evals   # strict type check
```

Copy `.env.example` to `.env` and fill in secrets before running anything
that talks to a database or LLM provider. Never commit `.env`.

## Knowledge source policy

`Knowledge source/` is the single immutable raw-corpus root (`docs/ADR/0010`
explains why it keeps this name rather than the design docs' spelling,
`knowledge_source/` — least-disruptive resolution of a real naming conflict
between the docs and the already-working repo). Never modified, renamed, or
rewritten — every file's SHA-256 is recorded in
`knowledge/00_registry/source_manifest.yaml` before anything reads it.
Proprietary/unverified-license PDFs and every archive are gitignored; only
the manifest, schemas, and generated OKF markdown are versioned. Large
archives (up to ~334 MB) are **not** physically duplicated into `knowledge/`
— disk headroom (`docs/ADR/0011`); the pipeline reads `original_path`
directly since it's already immutable.

## Level 3: real corpus, LS-DYNA parser, Structured + Hybrid RAG

**Source profiler** (`scripts/profile_knowledge_sources.py` →
`data/artifacts/source_profile.json`): recursively discovers every file
and archive member, hashes and classifies each. Last real run: 617 rows
(28 top-level, 589 archive members).

**Archive processing** (`packages/ingestion/archives.py`): safe ZIP/TAR/
TAR.GZ/TGZ member inspection — path-traversal, absolute-path,
decompression-bomb-ratio, duplicate-path, and symlink/hardlink guards —
without extracting to disk. `safe_extract_member()` extracts one member at
a time, re-validating independently.

**PDF pipeline** (`packages/ingestion/`): PyMuPDF-based (Docling deferred,
`docs/ADR/0011` — 12 GB free disk, `torch`/`transformers` too large a risk
right now). `qa.py`'s `build_extraction_report()` gives every PDF a
no-silent-loss report (per-page fault isolation, verified with real fault
injection) — every PDF gets `data/artifacts/extraction_reports/<id>.json`.
`structure.py` uses PyMuPDF's own `find_tables()`/`get_images()` for real
table/figure extraction (verified against the real corpus: 17 tables, 54
figures found in `UN_R94.pdf` alone) — persisted as `DocumentTable`/
`DocumentFigure` rows, not silently flattened into prose.

**LS-DYNA parser** (`packages/cae/lsdyna/`): a real lexer → parser →
include-graph resolver. Preserves every keyword's raw text, source file,
and line span, content-hashed; unknown keywords are never given invented
meaning. `PART`/`SECTION`/`MAT`/`CONTACT`/`CONTROL`/`DATABASE`/`INCLUDE`
get real field extraction into dedicated `cae_*` tables
(`packages/domain/cae.py`); `NODE`/`ELEMENT`/`BOUNDARY`/`CONSTRAIN`/
`DEFINE`/`PARAMETER` are detected/counted/raw-preserved generically
(`cae_keywords`) — matching `TRD_LEVEL3.md` §19's own schema, which has no
`cae_nodes`/`cae_elements` table. The include-graph resolves
`RESOLVED`/`MISSING`/`CYCLE`/`DUPLICATE`/`AMBIGUOUS`/`OUTSIDE_ROOT` by
basename match. Verified against real, messy production data: a 39-include
Honda Accord assembly deck resolves cleanly (a real bug — legitimate
`../../` navigation being wrongly refused — was found and fixed this way);
a Silverado deck genuinely reports `AMBIGUOUS` includes because its archive
has parallel `BASELINE`/`LIGHTWEIGHT` trees with identically-named files —
correctly *not* guessed.

**Structured + Hybrid RAG** (mandatory, `TRD_LEVEL3.md` §15):

```
BM25 (real BM25Okapi, rank-bm25)     Dense (pgvector, real semantic embeddings)
                    \                          /
                     v                        v
                       Reciprocal Rank Fusion
                                |
                                v
                Reranker (lexical + authority heuristic)
                                |
                                v
              Authority / relevance / dedup guard
                                |
                                v
                            Evidence
```

`packages/retrieval/structured.py` (deck/part/material/contact/control/
database/include queries) is **complementary**, not fused into this ranked
list (`PRD_LEVEL3.md` §14's own words; `docs/ADR/0012` records this as the
resolution of a real internal inconsistency between that document's own
prose and its architecture diagram).

Dense retrieval uses real semantic embeddings
(`sentence-transformers/all-MiniLM-L6-v2` via `fastembed`/ONNX Runtime —
`docs/ADR/0014`), chosen by `evals/embedding_benchmark.py` against the
hashing placeholder it replaced: MRR 0.838 vs 0.249 on the real golden set,
no `torch` dependency. Real measured comparison
(`evals/level3_hybrid_eval.py`, not tuned): BM25-only (MRR 0.917) and
Dense-only (MRR 0.838) are both individually strong, but naive RRF fusion
of the two actually *dilutes* to MRR 0.806 on this small (8-query) golden
set — a real, reported-as-measured RRF characteristic, not a bug. The
reranker recovers past both individual legs to MRR 0.938 and NDCG@10 0.954
(vs RRF-only's 0.852).

**OKF** (`knowledge/07_okf/`, `scripts/generate_okf_concepts.py`): one
curated concept file per ingested document (not one per section — that
would approach rewriting the whole manual), real extracted content
(truncated, never LLM-summarized), full source/page/section/authority
provenance in YAML frontmatter per `ENVIRONMENT_SETUP.md` §12's spec.

**Bounded, not unlimited-depth**: PDFs are ingested to a 20-page bound per
document (consistent with the pre-existing V1 precedent, `docs/ADR/0006`,
for an 8 GB RAM dev machine); CAE decks are parsed as their main/"combine"
file plus *direct* includes only (some individual component files in this
corpus are 100–170 MB) across 4 real vehicle/dummy-model families. Real
scale reached anyway: 46,765 real keyword rows, 4,992 parts, 1,724
materials, 99 contacts across 93 real LS-DYNA files — 2,085 of 2,098 real
parts (99.4%) have their material genuinely resolvable via structured
search within that bounded scope.

## Testing & evaluation

```powershell
uv run pytest                              # 191 tests, all against real fixtures/corpus where applicable
uv run python evals/scenario_eval.py       # numerical accuracy, all 10 synthetic scenarios
uv run python evals/retrieval_eval.py      # full-pipeline Recall@5/@10/MRR
uv run python evals/level3_hybrid_eval.py  # per-stage BM25/Dense/RRF/reranker comparison + NDCG@10
uv run python evals/embedding_benchmark.py # embedding candidate comparison (docs/ADR/0014)
uv run python evals/reranker_benchmark.py  # reranker candidate comparison (docs/ADR/0015)
```

## Limitations

- **Docling and OCR (tesseract) are not installed** — 12 GB free disk
  measured at decision time; Docling would resolve several GB of
  `torch`/`transformers`. Both are `Protocol`-based swap points
  (`docs/ADR/0011`) requiring no changes above their respective modules
  once installed.
- **A real cross-encoder reranker is implemented and benchmarked but not
  the production default** (`docs/ADR/0015`) — it genuinely improves
  ranking quality (NDCG@10 0.954 vs 0.938) but measured at ~3.5s/query on
  this CPU, too slow for an interactive Copilot turn. Available via
  `CrossEncoderReranker`/`get_cross_encoder_reranker()` for contexts that
  accept that cost.
- **Dense retrieval uses real semantic embeddings** as of `docs/ADR/0014`
  (`fastembed`/ONNX Runtime, no `torch`) — the earlier hashing placeholder
  is now the explicit "mock" tier for tests only, never production.
- **PDF ingestion is bounded to 20 pages/document**; CAE deck parsing is
  bounded to main + direct includes. Both are real, working, and tested at
  real scale — just not the entire 1.3 GB corpus at unlimited depth.
- **No CISS field data** — `Knowledge source/NHTSA_crash_test_field_data/`
  exists but the profiler found zero files there; recorded `NOT_AVAILABLE`
  in the manifest rather than assumed present.
- **No historical-case data, no chosen/benchmarked embedding model, no
  report generation, no mechanism/animation review, no production
  hardening** (auth, audit trail, a dedicated test database) — all
  unchanged from V1's own stated scope.
- Not a production deployment.
