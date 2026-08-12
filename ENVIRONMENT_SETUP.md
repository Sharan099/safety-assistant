# Passive Safety CAE Investigation Agent — Environment & Repository Setup

**Version:** 1.0  
**Platform:** Windows 10/11 x64  
**Development model:** Claude Code + Claude Sonnet  
**Python environment:** uv  
**Repository:** Git  
**Runtime LLM:** FreeLLMAPI behind an application provider interface

---

# 1. Hardware Constraints

Development machine:

```text
CPU: Intel Core i5-8250U
RAM: 8 GB
GPU: NVIDIA GeForce 940MX, 2 GB
OS: Windows x64
```

Design consequences:

- Prefer CPU-friendly tooling.
- Keep background services minimal.
- Do not require a local large language model.
- Do not run a local VLM over the entire corpus.
- Do not introduce Kubernetes, Kafka, Elasticsearch, or a separate vector database.
- Use PostgreSQL + pgvector.
- Use DuckDB + Parquet for analytical data.
- Use Docling's standard PDF pipeline first.
- Use VLM selectively only when document extraction quality requires it.
- Process the knowledge corpus incrementally rather than all at once.

---

# 2. Required Software

Install/check:

```text
Git
Python 3.12+
uv
Node.js LTS
Docker Desktop
Claude Code
```

Recommended verification:

```powershell
git --version
python --version
uv --version
node --version
npm --version
docker --version
docker compose version
claude --version
```

Do not install Conda.

Do not use global pip for project dependencies.

---

# 3. uv Strategy

`uv` is the single Python environment/package manager.

Create the project:

```powershell
uv init
```

Create the Python environment:

```powershell
uv python pin 3.12
uv venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies through:

```powershell
uv add <package>
```

Development dependencies:

```powershell
uv add --dev pytest pytest-asyncio ruff mypy pre-commit
```

Run commands through:

```powershell
uv run <command>
```

Examples:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format .
```

Never manually maintain a second `requirements.txt` as the source of truth.

The source of truth is:

```text
pyproject.toml
uv.lock
```

Commit both.

---

# 4. Recommended Python Dependencies

Install only what is required by the current phase.

## Core backend

```text
fastapi
uvicorn[standard]
pydantic
pydantic-settings
sqlalchemy
alembic
psycopg[binary]
pgvector
```

## Data/analysis

```text
numpy
pandas
scipy
pyarrow
duckdb
```

## Document ingestion

Start with:

```text
docling
pymupdf
```

Optional only if required after evaluation:

```text
docling[...]
ocr dependencies
VLM dependencies
```

Do not install every optional Docling model/runtime.

## Retrieval

```text
rank-bm25
```

Use PostgreSQL full-text search first where practical.

## Agent

```text
langgraph
langchain-core
```

Only add other LangChain packages when an actual integration needs them.

## LLM provider

Prefer an OpenAI-compatible client if FreeLLMAPI exposes that interface:

```text
openai
```

Keep the application interface provider-neutral.

## API/validation/testing

```text
httpx
pytest
pytest-asyncio
```

---

# 5. Node/Frontend

Create the frontend with Next.js + TypeScript.

Use:

```text
Next.js
TypeScript
Tailwind CSS
TanStack Query
```

For charts choose one library after evaluating the required interaction model:

```text
Plotly
OR
Apache ECharts
```

Do not install both.

---

# 6. Docker Services

For local infrastructure, initially run only:

```text
PostgreSQL + pgvector
```

Example `docker-compose.yml` concept:

```text
postgres
  image: pgvector-enabled PostgreSQL image
  port: 5432
  volume: postgres_data
```

Do not add Redis unless an actual background-job requirement appears.

Do not add MinIO until artifact handling requires object-storage semantics.

---

# 7. Git Initialization

From the repository root:

```powershell
git init
git branch -M main
```

Create `.gitignore` before the first commit.

At minimum exclude:

```text
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
node_modules/
.next/
.env
.env.*
!.env.example

data/raw/
data/artifacts/
knowledge/08_extracted/
knowledge/09_indexes/

*.pyc
*.log
.DS_Store
```

Important:

**Do not put proprietary CAE documents, internal reports, credentials, simulation artifacts, or licensed manuals into Git.**

The repository should contain:

```text
source manifest
metadata
schemas
code
tests
small synthetic fixtures
```

not proprietary source files.

---

# 8. Environment Variables

Create:

```text
.env.example
```

Example:

```text
DATABASE_URL=postgresql+psycopg://passive_safety:change_me@localhost:5432/passive_safety

LLM_PROVIDER=freellmapi
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=

KNOWLEDGE_ROOT=./knowledge
DATA_ROOT=./data

LOG_LEVEL=INFO
```

Never commit `.env`.

---

# 9. Repository Structure

Target:

```text
passive-safety-cae-agent/
│
├── PRD.md
├── TRD.md
├── APP_FLOW.md
├── UI_UX_DESIGN_BRIEF.md
├── BACKEND_SCHEMA.md
├── IMPLEMENTATION_PLAN.md
├── ENVIRONMENT_SETUP.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── apps/
│   ├── api/
│   └── web/
│
├── packages/
│   ├── domain/
│   ├── analysis/
│   ├── ingestion/
│   ├── retrieval/
│   └── agent/
│
├── knowledge/
│   ├── 00_registry/
│   ├── 01_regulations/
│   ├── 02_official_docs/
│   ├── 03_internal/
│   ├── 04_historical/
│   ├── 05_synthetic/
│   ├── 06_reference/
│   ├── 07_okf/
│   ├── 08_extracted/
│   └── 09_indexes/
│
├── data/
│   ├── raw/
│   ├── artifacts/
│   ├── parquet/
│   └── synthetic/
│
├── evals/
├── tests/
├── scripts/
└── docs/
    └── ADR/
```

---

# 10. Knowledge Architecture

Use Google's Open Knowledge Format (OKF) as the **canonical portable knowledge representation**.

OKF v0.2 is intentionally a directory of Markdown files with YAML frontmatter and emphasizes provenance, trust, lifecycle and attestation.

Reference:

https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf

The application database is NOT replaced by OKF.

Use:

```text
Original PDF
    ↓
Docling
    ↓
Structured intermediate representation
    ↓
OKF Markdown + YAML
    ↓
PostgreSQL metadata
    ↓
pgvector embeddings
```

This gives us:

- human-readable knowledge;
- Git-diffable knowledge;
- agent-readable knowledge;
- portable provenance;
- rebuildable indexes.

---

# 11. OKF Directory Design

Example:

```text
knowledge/07_okf/
├── index.md
├── regulations/
│   ├── index.md
│   ├── un-r16/
│   │   ├── index.md
│   │   └── requirements.md
│   ├── un-r94/
│   ├── un-r95/
│   └── un-r129/
│
├── solver/
│   ├── index.md
│   └── ls-dyna-r17/
│       ├── index.md
│       ├── contacts.md
│       ├── materials.md
│       ├── control-cards.md
│       ├── database-output.md
│       └── theory.md
│
├── historical/
└── synthetic/
```

Do not manually rewrite entire manuals into OKF.

Instead:

```text
PDF remains immutable source
OKF contains curated concepts/knowledge
each concept links to exact source/page/section
```

---

# 12. OKF Frontmatter

Use the official OKF specification fields where applicable.

Example concept:

```yaml
---
type: Concept
title: LS-DYNA Contact Definition
description: Conceptual explanation of the relevant contact formulation.
tags:
  - ls-dyna
  - contact
  - passive-safety
sources:
  - document_id: lsdyna-r17-vol1
    locator:
      page: 1234
      section: "..."
authority: official_documentation
status: verified
created_by: ingestion_pipeline
---
```

Do not invent fields that conflict with the OKF specification.

Project-specific metadata can be added only when necessary and documented.

---

# 13. Document Processing Strategy

Use a tiered pipeline.

## Stage A — deterministic parser

Use Docling for:

- PDF layout
- reading order
- tables
- formulas
- figures
- OCR when needed
- Markdown/JSON export

Docling supports advanced PDF understanding and structured document representation.

Reference:

https://github.com/docling-project/docling

## Stage B — extraction quality gate

Calculate:

```text
text coverage
page extraction success
table extraction success
OCR usage
layout confidence
missing-page detection
```

## Stage C — selective VLM

Only invoke a VLM for pages that fail the quality gate or contain difficult visual structures.

Candidate Docling VLM pipelines can be evaluated later.

Do NOT run a local VLM across the entire corpus on this 2 GB GPU.

## Stage D — human/agent review

Flag low-confidence pages.

Never silently discard extraction failures.

---

# 14. Image Handling

For each extracted figure/image:

```text
source document
revision
page
figure identifier
caption
bounding box
artifact URI
hash
```

Store images outside PostgreSQL.

Reference them from OKF/knowledge metadata.

---

# 15. Table Handling

Store:

```text
raw extraction
Markdown representation
structured JSON
page
bounding box
quality score
source hash
```

Tables must remain independently retrievable.

---

# 16. VLM Policy

VLM is a **fallback/enrichment tool**, not the default parser.

Use VLM for:

```text
complex diagrams
poorly structured tables
scanned pages
figures whose meaning is important
pages where deterministic extraction fails
```

Do not use VLM to invent missing engineering content.

Any VLM-generated interpretation must be labelled:

```text
AI_DERIVED
```

and linked to the source page/image.

---

# 17. RAG Architecture

Use:

```text
keyword retrieval
+
PostgreSQL FTS
+
dense vector retrieval
+
metadata filters
+
RRF
+
parent/child context expansion
```

Later benchmark:

```text
reranker
```

Do not add a reranker until the baseline is measured.

---

# 18. Retrieval Metadata

Every chunk must retain:

```text
source_type
authority_level
publisher
document
revision
page
section
chunk_id
parent_chunk_id
hash
```

---

# 19. Source Authority

Never flatten source authority.

Priority:

```text
REGULATION
OFFICIAL_DOCUMENTATION
INTERNAL_APPROVED
HISTORICAL
SYNTHETIC
LLM_REASONING
```

The UI must show the source category.

---

# 20. Testing Environment

Run:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Before committing:

```powershell
git status
git diff
```

---

# 21. First Commit

After repository bootstrap:

```powershell
git add .
git commit -m "chore: bootstrap passive safety cae agent"
```

Do not commit proprietary knowledge files.

---

# 22. Development Rules

Claude Code must:

- read all project Markdown contracts before implementation;
- make small commits;
- run tests after changes;
- never bypass failing tests;
- never fabricate unavailable knowledge;
- preserve provenance;
- keep deterministic analysis separate from LLM reasoning;
- document architectural decisions;
- avoid unnecessary dependencies;
- benchmark before replacing architecture.

---

# 23. Definition of Environment Ready

Environment is ready only when:

```text
uv environment works
✓ FastAPI imports
✓ PostgreSQL starts
✓ migration runs
✓ Next.js starts
✓ tests run
✓ lint runs
✓ Docling processes a sample PDF
✓ OKF sample validates against project rules
✓ Git repository initialized
✓ .env protected
✓ Claude Code reads CLAUDE.md
```
