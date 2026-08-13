# Passive Safety CAE Investigation Agent — Level 3 PRD

**Version:** 1.1  
**Status:** Implementation-ready  
**Milestone:** Level 3 — Research-grade, evidence-backed engineering system  
**Primary users:** Passive Safety / Occupant Protection / Crash / CAE Engineers  
**Implementation:** Claude Code + Claude Sonnet  
**Development hardware:** Windows PC, Intel i5-8250U, 8 GB RAM, NVIDIA GeForce 940MX 2 GB

---

# 1. Product Vision

Build a production-oriented engineering investigation workstation for passive-safety CAE engineers.

The product helps engineers investigate questions such as:

> Why did chest deflection increase between Run A and Run B?

> Which model configuration change is most strongly associated with the response change?

> Where did two signals first diverge?

> What does the applicable regulation require?

> What does the LS-DYNA model actually contain?

> What evidence supports or contradicts a proposed mechanism?

The application is **not a generic chatbot**.

The product principle is:

> **AI investigates. Deterministic code calculates. Authoritative sources establish requirements. Evidence supports conclusions. Engineers decide.**

---

# 2. Level-3 Objective

Level 3 upgrades the current V1 synthetic/document prototype into a research-grade system using authentic public engineering artifacts.

The system must work across:

```text
Regulatory / official knowledge
        +
Official solver documentation
        +
Public engineering reports
        +
Real public CAE artifacts
        +
Public test / field data
        +
Synthetic ground-truth cases
```

The application must preserve the distinction between:

```text
authoritative fact
parsed engineering fact
deterministic calculation
observation
hypothesis
inference
recommendation
unknown
```

---

# 3. Current V1 → Level 3

The existing V1 investigation workflow remains:

```text
Select Run A + Run B
        ↓
Quality Gate
        ↓
Comparability
        ↓
Configuration Diff
        ↓
Global Response Analysis
        ↓
Signal Analysis
        ↓
First Divergence
        ↓
Evidence Retrieval
        ↓
Hypothesis
        ↓
Engineer Review
```

Level 3 strengthens the data and evidence layer underneath this workflow.

---

# 4. Level-3 Data Families

The initial authentic public corpus consists of the downloaded artifacts under:

```text
knowledge_source/
```

Expected source families include:

```text
NHTSA Honda Accord 2014
NHTSA Honda Odyssey 2019 integrated seat belt
NHTSA adjustable sled buck
NHTSA structural countermeasure research
NHTSA public vehicle FE models
NHTSA THOR-05F
OpenRadioss ModelExchange
NHTSA CISS
Existing regulations / solver documentation
```

The actual corpus is determined by source profiling.

Do not assume a fixed number of files.

Synthetic cases remain separate:

```text
data/synthetic/
```

Synthetic cases are evaluation data, not authoritative engineering knowledge.

---

# 5. Immutable Source Principle

```text
knowledge_source/
```

is the canonical immutable raw source corpus.

Never modify source artifacts.

Never silently overwrite, rename, or rewrite the original:

```text
PDF
ZIP
TAR
.k
.key
.inc
CSV
SAS
```

Derived artifacts belong in:

```text
knowledge/
data/
```

The system must preserve SHA-256 provenance from original source → derived representation.

---

# 6. Core Product Requirements

## FR-01 Source discovery

Recursively discover:

```text
PDF
ZIP
TAR
TAR.GZ
TGZ
K
KEY
INC
CSV
SAS
TXT
MD
```

and additional files without assuming a fixed corpus.

Record:

```text
path
filename
extension
size
SHA-256
archive membership
source family
processing status
```

---

## FR-02 Archive processing

Archives must be inspected without modifying the original archive.

For every archive record:

```text
archive SHA-256
archive size
member path
member SHA-256
member size
compression
nested archive relationship
```

Extraction must protect against:

- path traversal;
- absolute paths;
- duplicate collisions;
- decompression bombs;
- unsafe links;
- excessive extraction size.

---

# 7. PDF Requirements

PDFs are first-class engineering knowledge sources.

The extraction system must preserve:

```text
text
headings
sections
tables
figures
captions
page numbers
page geometry where available
```

Primary structural extraction framework:

> **Docling**

OCR is a selective fallback.

VLM is an optional visual fallback.

---

# 8. PDF No-Silent-Loss Requirement

The system must not claim impossible "perfect extraction."

Instead it must guarantee:

> **No silent information loss.**

For every PDF:

```text
original page count
processed page count
pages with text
pages requiring OCR
pages with tables
pages with figures
pages with warnings
low-quality pages
failed pages
```

must be recorded.

A suspicious or failed page must result in:

```text
NEEDS_REVIEW
```

rather than silently disappearing.

---

# 9. PDF Visual QA

For suspicious pages:

```text
PDF
 ↓
render page
 ↓
inspect structural extraction
 ↓
OCR if necessary
 ↓
optional VLM
 ↓
validation result
```

Tables remain traceable to their source page.

Figures remain traceable to their source page.

Charts must not be converted into invented numeric values.

If numeric chart extraction cannot be validated:

```text
chart_present = true
numeric_data_validated = false
```

---

# 10. LS-DYNA First-Class Data Requirement

Most downloaded CAE archives contain LS-DYNA input decks.

The application must treat:

```text
.k
.key
.inc
```

as structured engineering artifacts, not just plain text documents.

The pipeline is:

```text
NHTSA / CAE ZIP or TAR
        ↓
archive inventory
        ↓
raw .k/.key/.inc files
        ↓
LS-DYNA parser
        ↓
structured engineering representation
        ↓
engineering entity search
        ↓
Hybrid RAG
```

---

# 11. LS-DYNA Keyword Requirements

At minimum detect and progressively parse:

```text
*PART
*MAT*
*SECTION*
*ELEMENT*
*NODE
*CONTACT*
*BOUNDARY*
*CONSTRAIN*
*CONTROL*
*DATABASE*
*DEFINE*
*PARAMETER
*INCLUDE
```

The parser must preserve:

```text
keyword
cards
raw text
source file
line start
line end
```

Unknown keywords must be preserved:

```text
status = UNKNOWN
raw_text = preserved
```

Never invent the engineering meaning of an unsupported keyword.

---

# 12. LS-DYNA Include Graph

The system must resolve:

```text
main.key
 ├── vehicle.k
 ├── dummy.k
 │    └── thor_head.k
 └── restraint.k
```

and record:

```text
resolved
missing
cycle
duplicate
outside_root
```

The system must distinguish:

```text
deck complete
deck incomplete
deck unresolved
```

---

# 13. Dataset Profiler

The first CAE processing stage is a profiler.

Example output:

| File | Type | Keywords | Includes | Parts | Materials | Sections | Contacts | Model hints |
|---|---|---|---:|---:|---:|---:|---:|---|
| Seat_Model_Explicit_TrackMAT.k | .k | PART, MAT, SECTION... | 4 | ... | ... | ... | ... | Seat |
| Rear_Impact_BioRid.key | .key | INCLUDE, CONTACT... | 7 | ... | ... | ... | ... | BioRID |
| Frontal_Impact_18deg_40kph.key | .key | INCLUDE... | 12 | ... | ... | ... | ... | THOR |

The profiler must generate this automatically from the actual corpus.

---

# 14. Structured CAE Search

The system must support structured queries against parsed engineering entities.

Examples:

```text
Find all CONTACT definitions in this deck.
Find all MAT definitions referenced by these PARTs.
Which PART uses this MATERIAL?
Which files are included by this model?
What DATABASE outputs are requested?
Which control cards are present?
```

Structured search is complementary to RAG.

---

# 15. Mandatory Structured + Hybrid RAG

**Hybrid RAG is a mandatory Level-3 requirement.**

The system must not rely on dense embeddings alone.

The retrieval architecture is:

```text
                         USER QUERY
                             │
                             ▼
                    Query Understanding
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
       BM25              Dense Search     Structured Search
    lexical search       pgvector          PostgreSQL
          │                  │                  │
          │                  │                  │
          └──────────────┬───┴──────────────────┘
                         ▼
               Reciprocal Rank Fusion
                         │
                         ▼
                    Candidate Set
                         │
                         ▼
                  Cross-Encoder
                    Reranker
                         │
                         ▼
              Authority / Metadata
                     Filtering
                         │
                         ▼
                  Relevance Gate
                         │
                         ▼
                 Evidence Builder
                         │
                         ▼
                   LangGraph Agent
```

---

# 16. BM25 / Keyword Retrieval

BM25 keyword search is mandatory.

It is particularly important for exact CAE identifiers:

```text
*MAT_024
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

The system must retain exact lexical matches.

PostgreSQL full-text / lexical search should be used initially rather than introducing Elasticsearch/OpenSearch solely for BM25.

---

# 17. Dense Retrieval

Dense retrieval uses embeddings stored in:

```text
pgvector
```

It is intended for:

```text
semantic similarity
paraphrasing
conceptual questions
engineering explanations
```

Embedding model/version must be recorded.

---

# 18. Structured Retrieval

Structured PostgreSQL retrieval must be used for parsed CAE entities:

```text
PART
MAT
SECTION
NODE
ELEMENT
CONTACT
BOUNDARY
CONTROL
DATABASE
INCLUDE
```

Example:

> Which materials are used by Part 1042?

This should be answered from structured data, not guessed from semantic similarity.

---

# 19. Reciprocal Rank Fusion

BM25, dense and structured candidates must be merged through a transparent fusion layer.

The initial implementation should use RRF.

Store:

```text
retriever
rank
score
fusion score
```

so retrieval can be evaluated.

---

# 20. Reranking

Use a reranker after fusion.

The reranker must not replace BM25/dense retrieval.

Pipeline:

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
```

The chosen reranker must be configurable and benchmarked on the user's CPU-constrained hardware.

---

# 21. Relevance / Grounding Gate

Do not pass arbitrary top-k results to the LLM.

Require:

```text
minimum relevance
authority compatibility
metadata compatibility
document scope
revision compatibility
```

If evidence is insufficient:

```text
INSUFFICIENT_EVIDENCE
```

The agent must not fabricate.

---

# 22. Authority Hierarchy

Use explicit authority metadata.

Example:

```text
E5 verified internal engineering evidence
E4 government / regulation
E3 official solver documentation
E2 public research
E1 synthetic benchmark
E0 LLM-generated
```

These levels are not automatically equivalent to legal authority. The actual source type and applicability must be preserved.

---

# 23. OKF

Use Google's Open Knowledge Format as a portable curated knowledge representation.

Pipeline:

```text
Original source
 ↓
Docling / LS-DYNA parser
 ↓
validated representation
 ↓
curated OKF concept
 ↓
PostgreSQL metadata
 ↓
pgvector
```

OKF is not the system database.

Every OKF concept must preserve:

```text
source
source SHA-256
document
revision
page / line
section
authority
```

---

# 24. Engineering Data Storage

Large simulation arrays should not be stored directly in PostgreSQL.

Use:

```text
Parquet
```

for normalized signals and large columnar data.

Use PostgreSQL for:

```text
metadata
relationships
features
provenance
retrieval metadata
```

---

# 25. Agent Requirements

The LLM is responsible for:

```text
intent understanding
investigation planning
tool selection
evidence synthesis
hypothesis generation
uncertainty explanation
next-step recommendation
```

Deterministic code is responsible for:

```text
quality gates
comparability
configuration diffs
numerical calculations
signal analysis
divergence
structured CAE queries
provenance
authorization
retrieval filtering
```

The LLM cannot bypass deterministic controls.

---

# 26. Agent Tooling

Existing tools remain the primary mechanism:

```text
run_quality_gate
assess_comparability
compare_configuration
analyze_signal
detect_first_divergence
retrieve_knowledge
retrieve_structured_cae
get_evidence
create_hypothesis
evaluate_evidence
request_engineer_review
```

The exact names should follow the existing repository.

---

# 27. Evaluation Requirements

Level 3 must evaluate every retrieval stage separately.

### BM25

```text
Recall@K
MRR
```

### Dense

```text
Recall@K
MRR
```

### Structured

```text
entity query correctness
relationship correctness
```

### RRF

```text
Recall@K
MRR
```

### Reranker

```text
NDCG@K
MRR
```

### Final evidence

```text
citation correctness
source correctness
authority correctness
```

### Agent

```text
tool selection accuracy
tool argument correctness
groundedness
unsupported claim rate
abstention accuracy
```

---

# 28. Golden Dataset

Create a Level-3 golden evaluation dataset containing:

```text
LS-DYNA keyword questions
LS-DYNA entity relationship questions
THOR questions
NHTSA model questions
regulatory questions
OpenRadioss questions
negative / unsupported questions
```

Every item must define:

```text
expected source
expected page/line where applicable
expected authority
acceptable concepts
expected retrieval path
```

---

# 29. Level-3 End-to-End Acceptance

A Level-3 investigation must demonstrate:

```text
Public source
 ↓
immutable source
 ↓
SHA-256
 ↓
archive inspection
 ↓
PDF/deck processing
 ↓
validation
 ↓
structured representation
 ↓
OKF
 ↓
PostgreSQL
 ↓
BM25
 ↓
Dense
 ↓
Structured search
 ↓
RRF
 ↓
Reranker
 ↓
Relevance gate
 ↓
LangGraph
 ↓
Deterministic tools
 ↓
Evidence-backed response
 ↓
Engineer review
```

---

# 30. Definition of Done

Level 3 is complete only when:

```text
Source integrity             PASS
Archive safety               PASS
PDF extraction               PASS
PDF completeness QA          PASS
LS-DYNA parser               PASS
Include graph                PASS
OKF                           PASS
PostgreSQL                   PASS
BM25 retrieval               PASS
Dense retrieval              PASS
Structured retrieval         PASS
RRF                           PASS
Reranker                      PASS
Grounding gate               PASS
Agent evaluation              PASS
End-to-end test               PASS
README                        UPDATED
```

No "Level 3 Ready" claim is allowed if a mandatory gate fails.

---

# 31. Non-Goals

Do not implement in Level 3:

- autonomous engineering sign-off;
- automatic regulatory approval;
- solver execution;
- automatic vehicle model modification;
- full enterprise RBAC;
- Kubernetes;
- multi-agent swarm;
- separate vector database;
- CFD expansion;
- full CarCrashNet ingestion before its data is officially accessible;
- automatic design optimization.
