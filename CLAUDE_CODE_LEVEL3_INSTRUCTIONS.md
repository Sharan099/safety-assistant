# CLAUDE CODE — LEVEL 3 IMPLEMENTATION INSTRUCTIONS

You are implementing Level 3 of the Passive Safety CAE Investigation Agent.

The existing V1 is already a working investigation workstation. Do not rebuild it.

Your objective is to make the existing application research-grade by connecting the real public engineering corpus under `knowledge_source/` to a validated ingestion, structured CAE parsing, **Structured + Hybrid RAG**, deterministic analysis and evaluated LangGraph investigation workflow.

---

# 1. NON-NEGOTIABLE RULE

**Do not start coding immediately.**

First inspect the repository and produce an implementation audit.

You must read:

```text
PRD.md
TRD.md
APP_FLOW.md
BACKEND_SCHEMA.md
UI_UX_DESIGN_BRIEF.md
IMPLEMENTATION_PLAN.md
ENVIRONMENT_SETUP.md
CLAUDE.md
README.md
PRD_LEVEL3.md
TRD_LEVEL3.md
```

Then inspect:

```text
knowledge/
knowledge_source/
data/
packages/
apps/
tests/
evals/
docs/ADR/
```

---

# 2. FIRST PHASE — AUDIT ONLY

Before modifying code, report:

```text
Current repository tree
Current ingestion architecture
Current PDF extraction
Current LS-DYNA handling
Current database schema
Current retrieval architecture
Current BM25/lexical search status
Current dense retrieval status
Current structured retrieval status
Current RRF status
Current reranker
Current LangGraph agent
Current deterministic analysis
Current tests
Current README
```

Then profile the actual corpus:

```text
number of PDFs
number of ZIPs
number of TAR/TAR.GZ/TGZ
number of .k
number of .key
number of .inc
number of CSV
number of SAS
other relevant files
```

Also inspect:

```text
nested archives
duplicate sources
existing extracted data
existing indexes
```

Do not assume there are 57 files.

Use the actual discovered count.

---

# 3. AUDIT REPORT

Before implementation provide:

```text
1. Current state
2. Target state
3. Architecture gaps
4. Data gaps
5. RAG gaps
6. Parser gaps
7. PDF extraction gaps
8. Database gaps
9. Test gaps
10. Files to modify
11. Files to create
12. Migration plan
13. Risks
14. Implementation phases
15. Acceptance tests
```

Do not silently reconcile conflicting architecture.

If two existing documents disagree, identify the conflict and choose the least disruptive solution consistent with Level 3 requirements.

---

# 4. SOURCE IMMUTABILITY

Never modify:

```text
knowledge_source/
```

No renaming.

No overwriting.

No editing.

No normalizing originals.

Derived artifacts only:

```text
knowledge/
data/
```

Every source must have SHA-256.

---

# 5. SOURCE PROFILER FIRST

Implement:

```text
scripts/profile_knowledge_sources.py
```

Output:

```text
data/artifacts/source_profile.json
data/artifacts/source_profile.parquet
```

Record all discovered files.

For archives, record members.

For LS-DYNA files, detect keyword names safely.

Do not perform full semantic parsing in the profiler.

---

# 6. ARCHIVE PROCESSING

Support:

```text
ZIP
TAR
TAR.GZ
TGZ
```

Requirements:

```text
safe extraction
path traversal protection
duplicate detection
member SHA-256
archive SHA-256
nested archive handling
size limits
restartability
```

Never execute files from an archive.

---

# 7. LS-DYNA DATASET PROFILER

This is a priority.

For every `.k/.key/.inc`, produce:

```text
file
type
keywords
include_count
PART count
MAT count
SECTION count
NODE count if reliably parsed
ELEMENT count if reliably parsed
CONTACT count
BOUNDARY count
CONTROL count
DATABASE count
model hints
```

The profiler must generate a machine-readable dataset.

---

# 8. LS-DYNA PARSER

Implement a reusable parser.

Minimum:

```text
*NODE
*ELEMENT*
*PART
*SECTION*
*MAT*
*CONTACT*
*BOUNDARY*
*CONSTRAIN*
*CONTROL*
*DATABASE*
*DEFINE*
*PARAMETER*
*INCLUDE
```

Preserve:

```text
keyword
raw text
line start
line end
source file
raw hash
parsed fields
parse status
```

Unknown keyword:

```text
UNKNOWN
```

but preserve raw content.

Never invent engineering meaning.

---

# 9. INCLUDE GRAPH

Build a graph for:

```text
*INCLUDE
```

Support:

```text
nested
missing
cycle
duplicate
outside-root
```

Produce a human-readable report and database representation.

---

# 10. PDF INGESTION

Use Docling as the primary structural parser.

Do not flatten PDFs immediately.

Preserve:

```text
pages
text
headings
sections
tables
figures
captions
```

---

# 11. PDF NO-SILENT-LOSS

For every PDF generate an extraction report.

Required:

```text
original page count
processed page count
text pages
OCR pages
table pages
figure pages
warnings
low-quality pages
failed pages
```

If a page cannot be confidently processed:

```text
NEEDS_REVIEW
```

Do not mark it complete.

---

# 12. OCR

OCR only where required.

Record:

```text
engine
version
page
confidence
```

Do not OCR every page by default.

---

# 13. VISUAL QA

Render suspicious pages.

Keep:

```text
original page
rendered page
extracted content
validation status
```

Optional VLM adapter can be used for:

```text
complex engineering diagrams
tables
charts
schematics
```

Every VLM result must be marked:

```text
VISUAL_INTERPRETATION
```

Never fabricate chart numbers.

---

# 14. OKF

Use OKF as a curated portable representation.

Do not convert the entire corpus into one giant Markdown document.

Use:

```text
knowledge/07_okf/
```

with source/page/line provenance.

Original source remains authoritative.

---

# 15. DATABASE

Inspect current schema first.

Use PostgreSQL + pgvector.

Do not add another database.

Add/extend entities for:

```text
source
document
page
section
table
figure
chunk
embedding
CAE deck
CAE file
keyword
include
part
material
section
contact
control
database
```

Use Alembic.

---

# 16. CRITICAL: STRUCTURED + HYBRID RAG

This is mandatory.

Do NOT implement dense-only RAG.

The retrieval system must contain three paths:

```text
1. BM25 / lexical
2. Dense / pgvector
3. Structured PostgreSQL CAE search
```

Then:

```text
BM25
+
Dense
+
Structured
 ↓
RRF
 ↓
Reranker
 ↓
Authority / metadata filter
 ↓
Relevance gate
 ↓
Evidence
```

---

# 17. BM25 / LEXICAL SEARCH

Implement PostgreSQL lexical retrieval.

Searchable content should include:

```text
PDF text
headings
tables
captions
LS-DYNA keyword names
LS-DYNA raw cards
OKF concepts
relevant metadata
```

BM25/exact lexical retrieval must be especially effective for:

```text
*MAT_024
*MAT_089
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE
*DATABASE_BINARY_D3PLOT
*PART
*SECTION
THOR-05F
HIC15
HIC36
UN R94
UN R16
Part ID
Material ID
```

If PostgreSQL's ranking is an approximation rather than exact BM25, document this honestly and test it.

Do not label a non-BM25 implementation as BM25.

---

# 18. DENSE SEARCH

Use pgvector.

Embeddings must be:

```text
versioned
rebuildable
batchable
```

Do not hard-code an embedding model into business logic.

---

# 19. STRUCTURED SEARCH

Implement SQL/ORM retrieval over parsed CAE entities.

Examples:

```text
Which material is used by Part 1042?
What contacts exist?
What sections are referenced?
What files are included?
Which control cards are present?
```

Do not ask the LLM to infer these relationships from text if structured data exists.

---

# 20. RRF

Implement Reciprocal Rank Fusion as deterministic code.

Record:

```text
candidate
BM25 rank
Dense rank
Structured rank
RRF score
```

The RRF implementation must be independently unit-tested.

---

# 21. RERANKER

Rerank the RRF candidate set.

Do not rerank before fusion unless an experiment proves a reason.

Record:

```text
candidate
reranker score
latency
model version
```

Choose a CPU-feasible model.

---

# 22. RELEVANCE GATE

No evidence enters the LLM merely because it is top-k.

Require:

```text
relevance
authority compatibility
metadata compatibility
revision compatibility
scope compatibility
```

If nothing passes:

```text
INSUFFICIENT_EVIDENCE
```

The LLM must not use its pretrained memory as a substitute.

---

# 23. RETRIEVAL TRACE

Every retrieval operation should make it possible to inspect:

```text
query
BM25 results
Dense results
Structured results
RRF ranking
Reranker ranking
final evidence
```

This is necessary for debugging and evaluation.

---

# 24. AGENT

Use the existing LangGraph architecture.

Do not create an unnecessary second agent framework.

The agent can decide:

```text
which tool
tool order
whether more evidence is required
whether to compare another signal
whether to retrieve documentation
whether competing hypotheses are needed
```

It cannot override deterministic tool outputs.

---

# 25. DETERMINISTIC BOUNDARY

The following remain deterministic:

```text
quality gates
comparability
configuration diff
signal statistics
peak
minimum
integral
correlation
first divergence
structured CAE queries
provenance
source hashes
retrieval filtering
```

The LLM interprets these results.

---

# 26. TESTING — DO NOT SKIP

Add/extend:

```text
unit tests
integration tests
regression tests
retrieval evaluation
agent evaluation
E2E tests
```

Run:

```powershell
uv run pytest
uv run ruff check .
uv run mypy apps packages scripts tests
```

---

# 27. BM25 EVALUATION

Create golden queries with exact engineering terms.

Examples:

```text
What does *DATABASE_BINARY_D3PLOT control?
What is *CONTACT_AUTOMATIC_SURFACE_TO_SURFACE?
Where is *MAT_024 used?
What does the THOR-05F qualification document specify?
```

Measure:

```text
Recall@K
MRR
```

---

# 28. DENSE EVALUATION

Create paraphrased queries.

Examples:

```text
Which LS-DYNA output database contains binary plot information?
How is automatic surface contact configured?
```

Measure:

```text
Recall@K
MRR
```

---

# 29. STRUCTURED EVALUATION

Test:

```text
part → material
part → section
deck → include
deck → contact
deck → control
```

Measure exact correctness.

---

# 30. HYBRID EVALUATION

Compare:

```text
BM25 only
Dense only
BM25 + Dense
BM25 + Dense + Structured
```

using the same golden set.

Measure:

```text
Recall@K
MRR
```

Do not claim improvement before running the experiment.

---

# 31. RERANKER EVALUATION

Measure:

```text
NDCG@K
MRR
latency
```

Compare:

```text
RRF ranking
vs
Reranked ranking
```

---

# 32. EVIDENCE EVALUATION

Measure:

```text
citation correctness
source correctness
page/line correctness
authority correctness
unsupported claim rate
abstention accuracy
```

---

# 33. PDF TESTS

Create/use fixtures:

```text
native
scanned
mixed
table-heavy
figure-heavy
multi-column
suspicious/malformed
```

Test:

```text
page accounting
text extraction
OCR fallback
tables
figures
warnings
failure visibility
```

---

# 34. LS-DYNA TESTS

Fixtures:

```text
simple.k
nested_include.key
unknown_keyword.k
malformed.k
comments.k
fixed_width.k
```

Test:

```text
keywords
cards
raw preservation
line spans
include graph
missing include
cycle
unknown keyword
```

---

# 35. REAL CORPUS SMOKE TEST

Select:

```text
one regulation PDF
one technical report
one mixed/scanned PDF if available
one ZIP containing .k
one standalone .k
one .key
one nested include deck
one TAR/TAR.GZ if available
```

Run:

```text
profile
hash
extract
parse
validate
index
retrieve
```

---

# 36. FULL CORPUS

Only after smoke tests pass:

```text
profile
→ hash
→ archive extraction
→ PDF validation
→ LS-DYNA parsing
→ OKF
→ PostgreSQL
→ BM25
→ dense
→ structured
→ RRF
→ reranker
→ agent
```

Use conservative concurrency.

---

# 37. FAILURE TESTS

Test:

```text
corrupt PDF
corrupt archive
missing include
unknown keyword
database unavailable
LLM unavailable
embedding unavailable
BM25 failure
structured search failure
reranker failure
OCR failure
VLM unavailable
zero retrieval
```

No silent failure.

---

# 38. README UPDATE

Update README with:

```text
Level 3 status
real corpus
knowledge_source policy
source provenance
PDF pipeline
LS-DYNA parser
OKF
Structured + Hybrid RAG
BM25
Dense
Structured search
RRF
Reranker
Grounding gate
agent
tests
evaluation
limitations
```

Do not claim production deployment.

---

# 39. ADRs

Create/update ADRs for:

```text
immutable knowledge_source
Docling primary PDF extraction
OCR fallback
optional VLM
LS-DYNA parser
PostgreSQL + pgvector
Parquet
OKF
Structured + Hybrid RAG
RRF
```

Only create an ADR where there is a meaningful architecture decision.

---

# 40. FINAL IMPLEMENTATION REPORT

At the end output:

```text
LEVEL 3 IMPLEMENTATION REPORT

Sources discovered:
PDF:
ZIP:
TAR:
K:
KEY:
INC:
Other:

Source integrity:
Archive:
PDF:
LS-DYNA:
OKF:
Database:

RAG:
BM25:
Dense:
Structured:
RRF:
Reranker:
Grounding gate:

Agent:
Tool selection:
Grounding:
Abstention:

Tests:
pytest:
ruff:
mypy:

E2E:
PASS / FAIL

Known failures:
Known limitations:

LEVEL 3:
READY / NOT READY
```

Never claim READY if a mandatory requirement is untested or failing.
