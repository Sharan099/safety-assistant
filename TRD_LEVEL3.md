# Passive Safety CAE Investigation Agent — Level 3 TRD

**Version:** 1.1  
**Status:** Technical implementation contract  
**Target:** Level 3 research-grade engineering system  
**Runtime:** Windows / Python / uv  
**Hardware:** Intel i5-8250U, 8 GB RAM, NVIDIA 940MX 2 GB

---

# 1. Mandatory Architecture

```text
knowledge_source/
      │
      ▼
Source Registry + SHA-256
      │
      ▼
Archive Processor
      │
      ├───────────────┐
      ▼               ▼
PDF Pipeline       CAE Pipeline
      │               │
  Docling          .k/.key/.inc
  OCR fallback       Parser
  Optional VLM       Include Graph
      │               │
      └───────┬───────┘
              ▼
       Validation Layer
              │
       ┌──────┴──────┐
       ▼             ▼
  knowledge/        data/
  OKF/docs          Parquet/features
       │             │
       └──────┬──────┘
              ▼
      PostgreSQL + pgvector
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
    BM25    Dense   Structured
      │       │        │
      └───────┼────────┘
              ▼
             RRF
              ▼
           Reranker
              ▼
      Authority/Metadata
          + Relevance
             Gate
              ▼
          Evidence
              ▼
          LangGraph
              ▼
     Deterministic Tools
              ▼
       Engineer Review
```

---

# 2. Architecture Constraint

Keep the existing:

```text
FastAPI
Next.js
PostgreSQL
pgvector
Alembic
LangGraph
Pydantic
uv
Docker Compose
Parquet
deterministic analysis package
```

Do not add:

```text
Elasticsearch/OpenSearch
Qdrant
Weaviate
Milvus
Neo4j
Redis
Kafka
Kubernetes
```

unless profiling demonstrates a concrete requirement and an ADR is approved.

PostgreSQL is the initial system of record, metadata store, lexical retrieval store and vector store.

---

# 3. Directory Contract

```text
project/
├── knowledge_source/        # immutable originals
├── knowledge/               # derived knowledge
│   ├── 00_registry/
│   ├── 07_okf/
│   ├── 08_extracted/
│   └── 09_indexes/
├── data/
│   ├── artifacts/
│   ├── parquet/
│   ├── features/
│   └── synthetic/
├── packages/
│   ├── ingestion/
│   ├── cae/
│   ├── analysis/
│   ├── retrieval/
│   ├── agent/
│   └── domain/
├── evals/
├── tests/
└── docs/ADR/
```

Adapt to the actual existing repository instead of duplicating packages.

---

# 4. Source Registry

Create/extend:

```text
knowledge/00_registry/source_manifest.yaml
```

Fields:

```yaml
source_id:
source_type:
category:
publisher:
title:
source_url:
local_path:
original_filename:
sha256:
size_bytes:
retrieved_at:
license:
validation_scope:
status:
```

Never invent:

```text
license
validation status
authority
source URL
```

Use:

```text
UNKNOWN
```

when unavailable.

---

# 5. Archive Processing

Support:

```text
.zip
.tar
.tar.gz
.tgz
```

Requirements:

- immutable original;
- SHA-256 before extraction;
- member inventory;
- member SHA-256;
- safe extraction;
- path traversal protection;
- duplicate path detection;
- extraction limits;
- nested archive support;
- failure isolation.

Output:

```text
data/artifacts/archive_manifests/<archive_id>.json
```

---

# 6. Dataset Profiler

Implement:

```text
scripts/profile_knowledge_sources.py
```

Outputs:

```text
data/artifacts/source_profile.json
data/artifacts/source_profile.parquet
```

Record:

```text
source_id
relative_path
filename
extension
size_bytes
sha256
archive_parent
archive_member
file_type
source_family
parser_candidate
status
```

For LS-DYNA:

```text
keyword_counts
include_count
part_count_if_known
material_count_if_known
section_count_if_known
node_count_if_known
element_count_if_known
contact_count_if_known
boundary_count_if_known
control_count_if_known
database_count_if_known
```

---

# 7. LS-DYNA Parser

Create or adapt:

```text
packages/cae/lsdyna/
├── lexer.py
├── parser.py
├── cards.py
├── models.py
├── profiler.py
├── include_graph.py
├── keyword_registry.yaml
└── schemas/
```

The parser must preserve:

```text
keyword
card
raw_text
source_file
line_start
line_end
raw_hash
```

---

# 8. LS-DYNA Keyword Registry

Initial high-value support:

```text
*NODE
*ELEMENT*
*PART
*SECTION*
*MAT*
*CONTACT*
*INCLUDE
*BOUNDARY*
*CONSTRAIN*
*CONTROL*
*DATABASE*
*DEFINE*
*PARAMETER*
```

Unknown keyword:

```yaml
status: UNKNOWN
```

Unknown content is still searchable raw content.

Never invent a keyword schema.

---

# 9. Include Graph

Represent:

```text
Deck
 └── Include
      └── Include
```

Statuses:

```text
RESOLVED
MISSING
CYCLE
DUPLICATE
OUTSIDE_ROOT
```

Store source line of each `*INCLUDE`.

Never execute arbitrary deck content.

---

# 10. PDF Pipeline

Primary:

```text
Docling
```

Intermediate representation must retain:

```text
document
page
text
heading
section
table
figure
caption
page metadata
```

Do not immediately flatten the PDF to plain text.

---

# 11. OCR Policy

OCR only when needed.

Trigger examples:

```text
no native text
low text coverage
image-only page
failed structural extraction
```

Record:

```text
ocr_used
ocr_engine
ocr_version
confidence
```

---

# 12. PDF QA

Every PDF receives:

```text
extraction_report.json
```

Required fields:

```text
source_sha256
original_page_count
processed_page_count
pages_with_text
pages_without_text
pages_ocr
pages_with_tables
pages_with_figures
pages_with_warnings
low_quality_pages
failed_pages
engine
engine_version
status
```

Statuses:

```text
PASS
PASS_WITH_WARNINGS
NEEDS_REVIEW
FAIL
```

---

# 13. No-Silent-Loss Rule

The pipeline must never silently omit:

```text
page
table
figure
section
OCR page
```

A missing page becomes a failed/needs-review condition.

The system must preserve the original PDF regardless of extraction result.

---

# 14. Visual QA

For suspicious pages:

```text
render page → inspect → OCR/VLM if needed → report
```

Store rendered page images outside PostgreSQL.

PostgreSQL stores metadata:

```text
document_id
page
image_uri
sha256
bbox
image_type
```

---

# 15. Optional VLM

Define an adapter:

```python
class VisualExtractionProvider(Protocol):
    def analyze_page(self, image_path: str) -> VisualExtractionResult: ...
```

Default:

```text
disabled
```

Use only for:

```text
complex tables
diagrams
schematics
charts
visual engineering relationships
```

VLM-derived content must be labeled:

```text
VISUAL_INTERPRETATION
```

Never present VLM interpretation as authoritative numeric measurement without validation.

---

# 16. Table Processing

Persist:

```text
table_id
document_id
page
caption
headers
rows
source coordinates
status
```

A failed table extraction must retain the source page.

---

# 17. Figure Processing

Persist:

```text
figure_id
document_id
page
caption
bbox
image_uri
sha256
```

Do not discard figures after text extraction.

---

# 18. OKF

Use:

```text
knowledge/07_okf/
```

for curated concepts.

Recommended categories:

```text
regulations/
solver/
cae/
occupant/
historical/
concepts/
```

Each concept must retain:

```text
source_id
source_sha256
document
revision
page/line
section
authority
derived_from
```

---

# 19. Database Schema

Inspect existing migrations first.

Extend rather than duplicate.

Required concepts:

```text
source_snapshots
documents
document_revisions
document_pages
document_sections
document_tables
document_figures
knowledge_chunks
knowledge_embeddings

cae_decks
cae_files
cae_keywords
cae_includes
cae_parts
cae_materials
cae_sections
cae_contacts
cae_controls
cae_databases
```

Use Alembic.

---

# 20. Retrieval Architecture — Mandatory

**This is a non-negotiable Level-3 requirement.**

Implement:

```text
                 QUERY
                   │
                   ▼
             Query normalization
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
      BM25       Dense       Structured
    PostgreSQL  pgvector     PostgreSQL
       │           │            │
       └───────────┼────────────┘
                   ▼
                  RRF
                   ▼
               Reranker
                   ▼
        Authority + metadata
               filtering
                   ▼
             Relevance gate
                   ▼
              Evidence
```

---

# 21. BM25 / Lexical Retrieval

Implement lexical search over searchable engineering text.

The searchable corpus must include, where appropriate:

```text
PDF text
headings
tables
captions
LS-DYNA keyword names
LS-DYNA raw cards
OKF concepts
source metadata
```

Important exact matches:

```text
*MAT_024
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE
*DATABASE_BINARY_D3PLOT
*PART
THOR-05F
HIC15
HIC36
UN R94
UN R16
Part ID
Material ID
```

PostgreSQL FTS is the initial lexical implementation.

If exact BM25 scoring is required by the implementation, document the PostgreSQL ranking approximation and benchmark it. Do not falsely claim a mathematically exact BM25 implementation if the actual function differs.

---

# 22. Dense Retrieval

Use:

```text
pgvector
```

Store:

```text
embedding
embedding_model
embedding_version
chunk_id
created_at
```

Embedding model must be configurable.

---

# 23. Structured Retrieval

Use SQL/ORM queries against parsed CAE entities.

Examples:

```text
find contacts in deck
find materials for part
find sections for parts
find include dependencies
find database outputs
find control cards
```

Structured retrieval must not be replaced by semantic similarity.

---

# 24. Reciprocal Rank Fusion

Implement RRF as a separate deterministic component.

Input:

```text
BM25 ranked results
Dense ranked results
Structured ranked results
```

Output:

```text
candidate_id
bm25_rank
dense_rank
structured_rank
rrf_score
```

This makes retrieval explainable and testable.

---

# 25. Reranker

Run the reranker after RRF.

The reranker must be configurable.

Benchmark CPU latency.

Do not use a model that makes the application unusable on the 8 GB machine.

---

# 26. Authority / Metadata Filtering

Filter candidates by:

```text
authority
source type
document
revision
solver
model
domain
applicability
```

Authority filtering occurs before final evidence selection.

---

# 27. Relevance Gate

If no candidate passes:

```text
INSUFFICIENT_EVIDENCE
```

The LLM receives the insufficiency state.

It must not fill the gap from model memory.

---

# 28. Retrieval Observability

Every retrieval request must record:

```text
query
BM25 candidates
dense candidates
structured candidates
RRF scores
reranker scores
final evidence
source IDs
latency
```

This is required for Level-3 evaluation.

---

# 29. Storage of Large CAE Data

Use:

```text
Parquet
```

for:

```text
signals
time histories
bulk normalized CAE data
features
```

Do not load full large datasets into memory.

Use:

```text
batching
streaming
lazy loading
```

---

# 30. Deterministic Analysis Boundary

LLM cannot calculate:

```text
peak
minimum
integral
correlation
time-to-peak
first divergence
configuration difference
quality metrics
```

These belong to deterministic Python analysis.

The LLM interprets their outputs.

---

# 31. Agent

Use the existing LangGraph architecture.

Agent responsibilities:

```text
understand request
choose tools
sequence investigation
request retrieval
generate hypotheses
compare evidence
explain uncertainty
recommend next step
```

Application responsibilities:

```text
tool authorization
data access
deterministic computation
source validation
retrieval filtering
provenance
```

---

# 32. Testing

Required test layers:

```text
unit
integration
regression
retrieval evaluation
agent evaluation
end-to-end
```

Run:

```powershell
uv run pytest
uv run ruff check .
uv run mypy apps packages scripts tests
```

Use repository-specific commands where different.

---

# 33. BM25 Tests

Create golden questions containing exact engineering identifiers.

Examples:

```text
What does *DATABASE_BINARY_D3PLOT control?
What does *CONTACT_AUTOMATIC_SURFACE_TO_SURFACE define?
Where is *MAT_024 used?
Which document discusses THOR-05F qualification?
```

Measure:

```text
Recall@K
MRR
```

---

# 34. Dense Retrieval Tests

Use paraphrased questions:

```text
What output database does LS-DYNA use for binary plot data?
Explain the surface-to-surface automatic contact definition.
```

Measure:

```text
Recall@K
MRR
```

---

# 35. Structured Retrieval Tests

Examples:

```text
Which material is assigned to Part 1042?
Which files are included by main.key?
Which contact definitions exist in this deck?
```

Measure:

```text
entity correctness
relationship correctness
```

---

# 36. RRF Tests

Compare:

```text
BM25 only
Dense only
Structured only
BM25 + Dense
BM25 + Dense + Structured
```

Measure:

```text
Recall@K
MRR
```

Do not claim that hybrid retrieval improves performance until measured.

---

# 37. Reranker Tests

Measure:

```text
NDCG@K
MRR
latency
```

Compare against the fused candidate ranking.

---

# 38. Final Evidence Tests

Measure:

```text
citation correctness
source correctness
page/line correctness
authority correctness
unsupported-answer rate
```

---

# 39. Agent Tests

Measure:

```text
tool selection accuracy
tool argument validity
evidence grounding
unsupported claim rate
abstention accuracy
hypothesis classification
```

---

# 40. PDF Tests

Fixtures:

```text
native text
scanned
mixed
table-heavy
figure-heavy
multi-column
malformed/suspicious
```

Verify:

```text
page accounting
text extraction
OCR fallback
table tracking
figure tracking
failure visibility
```

---

# 41. LS-DYNA Tests

Fixtures:

```text
simple.k
nested_include.key
unknown_keyword.k
malformed.k
comments.k
fixed_width.k
```

Verify:

```text
keyword detection
raw preservation
line spans
entity parsing
include resolution
missing includes
cycles
unknown keywords
```

---

# 42. Real Corpus Smoke Test

Select representative actual files:

```text
1 regulation PDF
1 technical report
1 scanned/mixed PDF if available
1 ZIP with .k files
1 standalone .k
1 .key
1 nested include deck
1 TAR/TAR.GZ if available
```

Run:

```text
discover
hash
profile
extract
parse
validate
index
retrieve
```

---

# 43. Full Corpus Test

Only after smoke tests pass:

```text
source profiling
→ hashing
→ archive manifests
→ extraction
→ PDF validation
→ LS-DYNA parsing
→ OKF
→ database
→ BM25
→ embeddings
→ structured retrieval
→ RRF
→ reranking
→ agent
```

Use conservative concurrency for the 8 GB machine.

---

# 44. Failure Tests

Test:

```text
corrupt PDF
missing include
corrupt ZIP
unknown keyword
LLM unavailable
embedding unavailable
database unavailable
BM25 unavailable
reranker unavailable
OCR failure
VLM unavailable
zero retrieval results
```

The system must fail explicitly.

---

# 45. Performance Constraints

The application must work on the development machine without requiring GPU inference.

Use:

```text
CPU-friendly models
batching
streaming
lazy loading
limited concurrency
incremental indexing
restartable ingestion
```

No full-corpus RAM loading.

---

# 46. README

README must document:

```text
Level 3 architecture
source corpus
knowledge_source policy
dataset profiler
PDF pipeline
LS-DYNA parser
OKF
BM25
dense retrieval
structured retrieval
RRF
reranker
relevance gate
agent
testing
evaluation
limitations
```

---

# 47. Definition of Done

Do not claim Level 3 READY unless:

```text
Source integrity        PASS
Archive processing      PASS
PDF extraction          PASS
PDF QA                  PASS
LS-DYNA parser          PASS
Include graph           PASS
OKF                     PASS
PostgreSQL              PASS
BM25                    PASS
Dense retrieval         PASS
Structured retrieval    PASS
RRF                     PASS
Reranker                PASS
Grounding gate          PASS
Agent evaluation        PASS
E2E                     PASS
pytest                  PASS
ruff                    PASS
mypy                    PASS
README                  UPDATED
```
