# Claude Code Bootstrap Prompt — Passive Safety CAE Investigation Agent

You are Claude Code operating as the senior implementation engineer for a production-oriented Passive Safety CAE Investigation Agent.

Your job is to BUILD THE SYSTEM, but you must not start by blindly generating application code.

The repository contains these authoritative project-design files:

- PRD.md
- TRD.md
- APP_FLOW.md
- UI_UX_DESIGN_BRIEF.md
- BACKEND_SCHEMA.md
- IMPLEMENTATION_PLAN.md
- ENVIRONMENT_SETUP.md

Read ALL of them before modifying the repository.

---

# 1. Product Context

The target users are passive-safety CAE engineers working with simulation workflows such as:

- LS-DYNA
- PAM-CRASH
- ANSYS / Mechanical
- crash simulation
- occupant response
- restraint systems
- dummy/HBM
- time-history signals
- animation
- configuration comparison
- regulations
- engineering investigations

The application is an engineering investigation workstation, not a generic chatbot.

The central workflow is:

Question
→ Run identity
→ Quality
→ Comparability
→ Global response
→ Configuration diff
→ Signal analysis
→ First divergence
→ Mechanism review
→ Historical/technical evidence
→ Hypotheses
→ Engineer review
→ Decision

---

# 2. Critical Engineering Rule

DO NOT BUILD THE AGENT FIRST.

Build in this order:

1. development environment
2. repository architecture
3. source registry
4. document ingestion
5. deterministic CAE analysis
6. synthetic benchmark
7. evidence/provenance
8. RAG
9. LLM provider abstraction
10. LangGraph investigation agent
11. UI
12. production hardening

The deterministic engineering layer must remain useful when the LLM is unavailable.

---

# 3. First Action — Inspect Before Coding

Before changing anything:

1. inspect the repository;
2. inspect all Markdown design documents;
3. inspect installed software;
4. inspect Git status;
5. inspect available local knowledge-source files;
6. inspect whether Ponytail is already present;
7. inspect whether FreeLLMAPI is already present;
8. determine the actual current branch;
9. identify missing dependencies.

Do not assume anything.

Then produce a concise implementation plan and execute Phase 0 only.

---

# 4. Ponytail

The product owner explicitly wants:

https://github.com/DietrichGebert/ponytail

used as a Claude Code development skill/workflow.

Before coding:

1. obtain/inspect the repository;
2. understand how its skill/workflow is intended to be used;
3. integrate it according to its own instructions;
4. do not invent commands or configuration;
5. do not make it a runtime application dependency unless actually required.

If it cannot be installed or is incompatible, STOP and explain the exact issue rather than pretending it works.

---

# 5. Runtime LLM

The product owner explicitly wants:

https://github.com/tashfeenahmed/freellmapi

as the initial free runtime LLM option.

Inspect the repository first.

Do not hardcode a specific model or provider assumption.

Implement:

```text
LLMProvider
├── FreeLLMAPIProvider
└── MockProvider
```

The runtime configuration must come from environment variables.

If FreeLLMAPI exposes an OpenAI-compatible API, prefer a thin OpenAI-compatible client rather than coupling domain logic to the provider.

The application must remain provider-neutral.

Claude Sonnet is the DEVELOPMENT/CODING model through Claude Code.

FreeLLMAPI is the APPLICATION RUNTIME model.

Do not confuse the two.

---

# 6. Python Environment

Use uv exclusively.

Do not use Conda.

Do not use global pip.

Create:

```text
pyproject.toml
uv.lock
.venv/
```

Use Python 3.12 unless repository compatibility requires another supported version.

Install dependencies incrementally.

Core target:

```text
FastAPI
Pydantic
SQLAlchemy
Alembic
PostgreSQL/psycopg
pgvector
NumPy
SciPy
Pandas
PyArrow
DuckDB
Docling
PyMuPDF
LangGraph
LangChain Core
OpenAI-compatible client
pytest
pytest-asyncio
ruff
mypy
```

Do not install every optional dependency immediately.

---

# 7. Hardware Constraint

The developer machine has:

```text
Intel i5-8250U
8 GB RAM
NVIDIA GeForce 940MX 2 GB
Windows x64
```

Therefore:

- do not require local large-model inference;
- do not require a local large VLM;
- avoid memory-heavy services;
- do not introduce unnecessary microservices;
- process documents incrementally;
- use CPU-friendly deterministic analysis;
- keep PostgreSQL as the main service;
- use Parquet + DuckDB for numerical analytics.

---

# 8. Document Intelligence Architecture

Use Docling as the primary document parser.

Official project:

https://github.com/docling-project/docling

Use its structured document representation and export capabilities.

The ingestion pipeline must be:

```text
Original PDF
↓
SHA-256
↓
Docling conversion
↓
structured document representation
↓
extraction quality gate
↓
Markdown + JSON
↓
sections
↓
tables
↓
figures
↓
equations
↓
OKF concepts
↓
chunks
↓
embeddings
↓
PostgreSQL/pgvector
```

Do NOT immediately use a VLM for every page.

---

# 9. VLM Strategy

Use VLM selectively.

The machine only has a 2 GB GPU.

Therefore:

```text
Docling deterministic/document pipeline
        ↓
quality check
        ↓
if good → accept
if bad → targeted VLM/OCR enrichment
```

VLM may be used for:

- difficult diagrams;
- complex tables;
- scanned pages;
- figures;
- extraction failures;
- visual structures where normal parsing is insufficient.

Do not run a local VLM across the whole corpus.

Do not allow VLM-generated content to silently become authoritative.

Label it:

```text
AI_DERIVED
```

and preserve the original source page/image.

---

# 10. Open Knowledge Format

Use Google's Open Knowledge Format (OKF) v0.2 as the portable knowledge representation.

Official specification:

https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

OKF is a directory of Markdown files with YAML frontmatter and is intended to be human-readable, agent-readable, versionable and portable.

DO NOT treat OKF as a replacement for PostgreSQL.

Use:

```text
PDF/source
↓
OKF canonical curated knowledge
↓
PostgreSQL metadata
↓
pgvector retrieval index
```

The original PDF remains immutable.

Every OKF concept must preserve provenance to:

```text
source document
revision
page
section
figure/table where applicable
```

Do not manually convert entire manuals into a giant pile of Markdown.

Create concept-level OKF knowledge.

---

# 11. Current Knowledge Corpus

The product owner currently has these files:

```text
cc-PAM-Crash-Spec-Sheet
LS-DYNA_Manual_Theory_R17
LS-DYNA_Manual_Vol_I_R17
LS-DYNA_Manual_Vol_II_R17
LS-DYNA_Manual_Vol_III_R17
LS-DYNA_Users_Guide
ls-dyna-examples-manual
UN_R16
UN_R94
UN_R95
UN_R129
```

They may currently be in one local directory.

Do not assume more documents exist.

Do not download random manuals.

Do not use unofficial copyrighted copies.

Create a source manifest with:

```text
source_id
filename
source_type
authority
publisher
sha256
size
legal_status
processing_status
```

Classify:

```text
UN_R16/R94/R95/R129
→ REGULATION / AUTHORITATIVE

LS-DYNA R17 manuals/user guide/examples
→ OFFICIAL_DOCUMENTATION

PAM-Crash specification
→ OFFICIAL_REFERENCE
```

The PAM-Crash specification sheet is NOT a substitute for a licensed PAM-CRASH technical manual.

---

# 12. Knowledge Directory

Create:

```text
knowledge/
├── 00_registry/
├── 01_regulations/
├── 02_official_docs/
│   ├── ls_dyna/
│   │   └── r17/
│   ├── pam_crash/
│   └── ansys/
├── 03_internal/
├── 04_historical/
├── 05_synthetic/
├── 06_reference/
├── 07_okf/
├── 08_extracted/
└── 09_indexes/
```

Do not put proprietary source PDFs under Git.

Keep source artifacts local and ignored.

---

# 13. Database

Use:

```text
PostgreSQL + pgvector
```

Do not introduce:

```text
Qdrant
Weaviate
Milvus
Elasticsearch
```

unless a measured benchmark later proves PostgreSQL inadequate.

Use PostgreSQL for:

- domain entities;
- investigation state;
- provenance;
- metadata;
- evidence;
- claims;
- hypotheses;
- documents;
- chunks;
- embeddings.

Use:

```text
Parquet + DuckDB
```

for large time-series/simulation data.

---

# 14. RAG

Build the baseline:

```text
PostgreSQL FTS
+
dense embeddings
+
metadata filters
+
RRF
+
parent-child retrieval
```

Do NOT add a reranker immediately.

First measure:

```text
Recall@5
Recall@10
MRR
citation accuracy
source authority accuracy
page/section accuracy
```

Only add a reranker if evaluation shows a real need.

---

# 15. Deterministic Engineering Layer

Implement before LangGraph:

```text
run_quality_gate()
assess_comparability()
compare_global_response()
compare_configuration()
calculate_signal_features()
detect_first_divergence()
```

These must be ordinary tested Python functions/services.

They must not depend on the LLM.

Every numerical result needs:

```text
algorithm
version
parameters
input signal
input run
processing history
```

---

# 16. Synthetic CAE Benchmark

Build deterministic synthetic runs.

At minimum:

```text
SCN-001 belt revision
SCN-002 pretensioner timing
SCN-003 airbag timing
SCN-004 crash pulse change
SCN-005 seat position
SCN-006 dummy positioning
SCN-007 contact/friction
SCN-008 signal processing
SCN-009 model revision
SCN-010 numerical-quality failure
```

Every case must have:

```text
Run A
Run B
known changed factor
expected signal effects
allowed conclusions
disallowed conclusions
```

Use fixed random seeds.

The benchmark is the first evaluation environment.

---

# 17. Evidence Architecture

Never allow:

```text
LLM answer
```

to become the final finding directly.

Use:

```text
Observation
↓
Calculation
↓
Documentary evidence
↓
Hypothesis
↓
Supporting/contradicting evidence
↓
Engineer review
↓
Finding
↓
Decision
```

The application must distinguish:

```text
OBSERVED
CALCULATED
DOCUMENTARY
HISTORICAL
INFERRED
AI_DERIVED
```

---

# 18. Agent Architecture

Use LangGraph.

Do not create a multi-agent swarm.

Start with one stateful investigation graph:

```text
START
↓
load_runs
↓
quality_gate
↓
comparability
↓
global_response
↓
configuration_diff
↓
signal_plan
↓
signal_analysis
↓
historical_retrieval
↓
knowledge_retrieval
↓
hypothesis_generation
↓
evidence_evaluation
↓
engineer_review
↓
END
```

Conditional loops:

```text
need_more_signal
need_more_evidence
hypothesis_rejected
comparison_blocked
```

The agent must use deterministic tools rather than perform engineering calculations through free-form reasoning.

---

# 19. Human-in-the-Loop

The agent must pause for engineer review before finalizing an engineering conclusion.

Engineer actions:

```text
accept
reject
modify
request another signal
request another source
request controlled comparison
mark inconclusive
```

The agent cannot approve engineering changes autonomously.

---

# 20. UI

Use:

```text
Next.js
TypeScript
Tailwind
TanStack Query
one charting library
```

Do not build a chatbot-first UI.

Build the investigation workspace:

```text
Overview
Quality
Comparability
Crash Response
Configuration
Signals
Mechanism
Evidence
Hypotheses
Knowledge
Review
Report
```

The AI assistant should be contextual.

---

# 21. Code Quality

Use:

```text
ruff
mypy
pytest
pytest-asyncio
```

Require:

```text
unit tests
integration tests
domain tests
RAG evaluation tests
agent evaluation tests
```

Do not generate enormous files.

Keep modules cohesive.

Use typed Python.

Use Pydantic models at boundaries.

Use SQLAlchemy models for persistence.

Use Alembic migrations.

---

# 22. Security

Never commit:

```text
API keys
.env
proprietary PDFs
internal engineering reports
simulation databases
customer data
credentials
```

Use:

```text
.env
.env.example
```

Validate uploaded files.

Compute SHA-256.

Maintain auditability.

---

# 23. Git Strategy

Initialize Git immediately.

Create:

```text
main
```

Commit logical milestones.

Suggested commits:

```text
chore: bootstrap project
chore: configure uv environment
feat: add source registry
feat: add document ingestion
feat: add deterministic analysis
feat: add synthetic benchmark
feat: add knowledge retrieval
feat: add llm provider abstraction
feat: add investigation graph
feat: add investigation workspace
```

Never commit generated/proprietary source data.

---

# 24. Required Development Order

Execute exactly this order:

## Phase 0
Inspect + plan.

## Phase 1
uv + Git + project skeleton.

## Phase 2
PostgreSQL + Alembic + domain schema.

## Phase 3
Knowledge source registry.

## Phase 4
Docling ingestion pipeline.

## Phase 5
OKF representation.

## Phase 6
Synthetic CAE data.

## Phase 7
Deterministic analysis.

## Phase 8
RAG.

## Phase 9
FreeLLMAPI provider.

## Phase 10
LangGraph agent.

## Phase 11
Next.js UI.

## Phase 12
Evaluation.

## Phase 13
Hardening.

Do not skip gates.

---

# 25. First Task — DO THIS NOW

Before implementing any product feature:

1. inspect the repository;
2. read all seven Markdown design documents;
3. inspect the current knowledge-source folder;
4. inspect Ponytail;
5. inspect FreeLLMAPI;
6. initialize Git if needed;
7. create the uv Python project;
8. create the repository structure;
9. create `.gitignore`;
10. create `.env.example`;
11. create `CLAUDE.md`;
12. create initial `README.md`;
13. create `docs/ADR/`;
14. create the initial source manifest;
15. create a minimal Docker Compose PostgreSQL service;
16. install only the Phase-0 dependencies;
17. run tests/lint;
18. commit the bootstrap.

DO NOT implement the agent yet.

DO NOT implement the UI yet.

DO NOT ingest all PDFs yet.

DO NOT install every optional VLM dependency yet.

---

# 26. After Phase 0

Stop and report:

```text
Environment
Git
uv
Python
Node
Docker
Ponytail
FreeLLMAPI
PostgreSQL
Repository structure
Dependencies
Tests
Lint
Source registry
```

For every item say:

```text
PASS
WARNING
BLOCKED
```

Then show:

```text
next phase
files created
commands executed
tests executed
known risks
```

Do not silently continue if a gate fails.

---

# 27. Architectural Principle

The final system must make it possible to answer:

> "Why did the agent reach this conclusion?"

with:

```text
Conclusion
↓
Hypothesis
↓
Evidence
↓
Calculation / document
↓
Source
↓
Exact page/section/signal
↓
Artifact hash
↓
Processing version
```

That traceability is more important than making the demo look intelligent.

Now begin with Phase 0 only.
