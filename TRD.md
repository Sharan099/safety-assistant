# Passive Safety CAE Investigation Agent — Technical Requirements Document

**Version:** 2.0  
**Status:** V1 Technical Contract  
**Development model:** Claude Code + Claude Sonnet  
**Runtime LLM:** FreeLLMAPI through an abstraction layer  
**Primary database:** PostgreSQL + pgvector  
**Large data:** Parquet/HDF5  
**Analytics:** DuckDB  
**Primary solver knowledge:** LS-DYNA R17 official documents currently available locally

---

# 1. Architecture

```text
                         Web UI
                           │
                           ▼
                       FastAPI
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
         Domain API    Analysis API   Knowledge API
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                  Investigation Service
                           │
                       LangGraph
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
 Deterministic        Retrieval           LLM Runtime
 Analysis Tools       Services            FreeLLMAPI
       │                   │                   │
       ▼                   ▼                   ▼
  Parquet/DuckDB    PostgreSQL/pgvector   Provider APIs
       │                   │
       └──────────┬────────┘
                  ▼
            Evidence Store
                  │
                  ▼
             PostgreSQL
```

---

# 2. Technology Stack

## Backend

```text
Python 3.12+
FastAPI
Pydantic v2
SQLAlchemy 2
Alembic
PostgreSQL
pgvector
DuckDB
PyArrow
Parquet
```

## Frontend

```text
Next.js
TypeScript
React
Tailwind CSS
TanStack Query
Plotly or Apache ECharts
```

Choose only one charting library for V1.

---

# 3. Agent Framework

Use:

```text
LangGraph
```

Reason:

- stateful investigation
- conditional routing
- human review
- tool calls
- loops
- persistence

Architecture:

```text
One Investigation Agent
+
specialized deterministic tools
```

Do not create a swarm of autonomous agents.

---

# 4. Runtime LLM

Use a provider abstraction:

```text
LLMProvider
├── FreeLLMAPIProvider
├── ClaudeProvider (optional/future)
└── MockProvider
```

The application must not depend directly on a single provider.

FreeLLMAPI is the first development runtime.

The exact model/provider available through FreeLLMAPI must be configurable and verified at runtime.

Do not hardcode assumptions about a specific free model.

---

# 5. Claude Code / Ponytail

Before implementation:

1. obtain the Ponytail repository;
2. inspect its skill structure;
3. install/use the relevant skill according to its own documentation;
4. validate that Claude Code can discover/use the skill;
5. record the chosen workflow in `CLAUDE.md` or project development docs.

Repository supplied by the user:

`https://github.com/DietrichGebert/ponytail`

Ponytail is a development-time dependency.

It is not a production runtime dependency unless a concrete runtime use case is discovered.

---

# 6. Hardware Constraints

Target development device:

```text
CPU: Intel i5-8250U
RAM: 8 GB
GPU: NVIDIA GeForce 940MX 2 GB
OS: Windows
```

Therefore:

```text
No local large LLM requirement
No 30B+ inference
No Kubernetes
No Kafka
No Elasticsearch
No unnecessary microservices
No separate vector database
```

Use:

```text
PostgreSQL + pgvector
DuckDB
Parquet
local filesystem
Docker where useful
```

Batch processing is preferred.

---

# 7. Database Decision

Use **PostgreSQL + pgvector**.

Reason:

- relational domain model
- JSONB for engineering metadata
- full-text search
- vector search
- transactions
- mature tooling
- fewer services
- low operational complexity
- suitable for an 8 GB development machine

Do not introduce Qdrant/Weaviate/Milvus in V1.

If pgvector becomes a measured bottleneck, benchmark alternatives before changing architecture.

---

# 8. Large Time-Series Storage

Do not store raw CAE time histories as PostgreSQL rows.

Use:

```text
Parquet
```

for columnar signal data.

Use:

```text
DuckDB
```

for analytical queries.

Example:

```text
signals/
└── RUN-0041/
    ├── metadata.parquet
    ├── occupant.parquet
    ├── restraint.parquet
    └── vehicle.parquet
```

Actual storage format may later use HDF5 where required by source data.

---

# 9. Artifact Storage

For V1:

```text
D:\PassiveSafetyAI\data\artifacts\
```

Later:

```text
MinIO-compatible object storage
```

Store:

```text
PDF
images
animation
CAE files
reports
large databases
```

PostgreSQL stores references and hashes.

---

# 10. Knowledge Source Structure

Current local corpus:

```text
Knowledge source/
├── cc-PAM-Crash-Spec-Sheet
├── LS-DYNA_Manual_Theory_R17
├── LS-DYNA_Manual_Vol_I_R17
├── LS-DYNA_Manual_Vol_II_R17
├── LS-DYNA_Manual_Vol_III_R17
├── LS-DYNA_Users_Guide
├── ls-dyna-examples-manual
├── UN_R16
├── UN_R94
├── UN_R95
└── UN_R129
```

These are currently stored in one folder.

The ingestion pipeline must move/copy them into the canonical structure without modifying originals.

Recommended:

```text
knowledge/
├── 00_registry/
├── 01_regulations/
│   └── unece/
│       ├── UN_R16/
│       ├── UN_R94/
│       ├── UN_R95/
│       └── UN_R129/
│
├── 02_official_docs/
│   ├── ls_dyna/
│   │   └── r17/
│   ├── pam_crash/
│   └── ansys/
│
├── 03_internal/
├── 04_historical/
├── 05_synthetic/
├── 06_reference/
├── 07_okf/
├── 08_extracted/
└── 09_indexes/
```

---

# 11. Current Source Classification

```text
UN_R16
UN_R94
UN_R95
UN_R129
→ AUTHORITATIVE / REGULATION

LS-DYNA R17 Vol I
LS-DYNA R17 Vol II
LS-DYNA R17 Vol III
LS-DYNA Theory R17
LS-DYNA User's Guide
LS-DYNA Examples
→ OFFICIAL_DOCUMENTATION

cc-PAM-Crash-Spec-Sheet
→ OFFICIAL_REFERENCE / PAM_CRASH
```

Do not classify the public PAM specification as a full technical manual.

---

# 12. Missing Source Categories

The system must explicitly represent:

```text
NOT_AVAILABLE
NOT_AUTHORIZED
NOT_INGESTED
```

Current gaps:

```text
LS-DYNA Database Manual
Licensed PAM-CRASH manuals
Additional ANSYS Mechanical docs
Internal reports
Historical investigations
```

Do not fill these gaps with hallucinated content.

---

# 13. Knowledge Ingestion Pipeline

```text
Original file
   ↓
Source registration
   ↓
SHA-256
   ↓
File inspection
   ↓
Text/layout extraction
   ↓
Extraction quality gate
   ↓
Page representation
   ↓
Section detection
   ↓
Table extraction
   ↓
Figure extraction
   ↓
Equation extraction where useful
   ↓
Canonical Markdown
   ↓
Semantic chunks
   ↓
Metadata
   ↓
Embeddings
   ↓
PostgreSQL + pgvector
```

---

# 14. PDF Extraction Strategy

Start with:

```text
PyMuPDF
```

because it is lightweight and suitable for the local machine.

Use OCR only when:

- text layer is missing;
- text quality is poor;
- page is scanned;
- extraction quality fails.

Do not OCR every page by default.

---

# 15. Images

For each useful figure:

```text
document_id
revision_id
page
figure_number
caption
image_uri
bounding_box
```

Store image separately.

The chunk can contain:

```text
[FIGURE: document_figure_id]
caption text
```

The image itself remains in artifact storage.

---

# 16. Tables

Store:

```text
table metadata
markdown representation
JSON representation
image crop
page
bounding box
extraction method
quality score
```

Tables must not be flattened into ordinary prose only.

---

# 17. Equations

Where practical store:

```text
LaTeX
image
page
equation number
```

Do not attempt to infer engineering meaning from an equation with an uncertain extraction.

---

# 18. Chunking

Use structure-aware chunking.

Preferred hierarchy:

```text
Document
→ Chapter
→ Section
→ Subsection
→ Paragraph/table/figure
```

Do not blindly split every PDF into fixed-size chunks.

Each chunk must retain:

```text
document
revision
page
section
source locator
chunk type
```

---

# 19. Retrieval

V1:

```text
PostgreSQL FTS
+
pgvector
+
metadata filters
+
RRF
+
parent-child expansion
```

Retrieval filters:

```text
source_type
authority_level
document
revision
solver
version
regulation
section
```

---

# 20. Embeddings

Do not commit to an embedding model before benchmarking.

Create an interface:

```text
EmbeddingProvider
```

Candidate models may be tested based on:

- CPU speed
- RAM usage
- embedding quality
- dimension
- licensing
- multilingual support

Store:

```text
model_name
model_version
dimensions
created_at
```

---

# 21. Reranking

Reranking is optional in V1.

First benchmark:

```text
BM25
+
dense
+
RRF
```

If retrieval quality requires reranking, add a lightweight reranker.

Do not introduce a slow local reranker simply because it is popular.

---

# 22. RAG Source Priority

For a regulatory question:

```text
Regulation
>
Official documentation
>
Internal
>
Historical
>
Synthetic
>
LLM
```

For a solver implementation question:

```text
Official solver documentation
>
Internal approved documentation
>
Historical
>
General reference
>
LLM
```

For a historical precedent:

```text
Historical
>
Internal
>
Synthetic
>
General documentation
>
LLM
```

---

# 23. Agent Tool Layer

Initial tools:

```text
load_run
get_run_manifest
run_quality_gate
assess_comparability
compare_global_response
compare_configuration
select_signal_plan
analyze_signal
detect_first_divergence
retrieve_historical_cases
retrieve_knowledge
get_source
create_hypothesis
evaluate_evidence
request_engineer_review
create_follow_up
```

Tools return structured JSON/Pydantic objects.

---

# 24. Agent State

```text
InvestigationState:
    investigation_id
    question
    run_a
    run_b
    primary_metric
    quality
    comparability
    global_response
    configuration_diff
    signal_plan
    signal_results
    events
    historical_cases
    knowledge_evidence
    hypotheses
    contradictions
    unknowns
    review
    decision
```

---

# 25. Guardrails

The agent must not:

- invent sources;
- invent page numbers;
- invent numerical values;
- invent regulatory limits;
- claim a document was consulted if it was not retrieved;
- claim causality from temporal precedence alone;
- ignore contradictory evidence;
- override engineer review;
- use LLM-generated text as authoritative evidence.

---

# 26. Observability

V1 logging:

```text
request_id
investigation_id
agent_step
tool_name
tool_duration
source_ids
model/provider
token usage if available
errors
```

Later:

```text
OpenTelemetry
Prometheus
Grafana
```

Do not add them before core functionality works unless needed for debugging.

---

# 27. Security

Minimum:

- environment variables for secrets
- no API keys in Git
- path validation
- file size limits
- file type validation
- source hash
- audit trail for engineer decisions

---

# 28. Deployment

V1 target:

```text
Local Windows development
```

Later:

```text
Docker
→ VPS/cloud
→ object storage
→ PostgreSQL
```

Do not design cloud infrastructure before the local product is validated.

---

# 29. OKF / Portable Knowledge

If the project uses Google Open Knowledge Format-compatible Markdown/YAML, it should be treated as a **portable curated knowledge representation**, not as the primary database.

Use:

```text
OKF-style Markdown/YAML
→ curated knowledge
PostgreSQL
→ application system of record
pgvector
→ retrieval
Parquet
→ numerical data
```

The application should be able to regenerate indexes from the canonical knowledge files.

---

# 30. Runtime LLM Failure Handling

If FreeLLMAPI is unavailable:

```text
Agent run
→ detect provider failure
→ preserve investigation state
→ show unavailable
→ allow retry
```

Do not silently substitute an unknown model.

A deterministic analysis should remain usable without an LLM.

---

# 31. Technical Non-Goals

No:

```text
microservice explosion
Kafka
Kubernetes
Elasticsearch
separate vector DB
local giant LLM
autonomous solver execution
```

until measured requirements justify them.
