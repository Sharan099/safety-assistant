# Passive Safety CAE Investigation Agent — Level 3 Final Working Fix

**Version:** 1.0  
**Purpose:** Final implementation contract for overcoming the current Level-3 limitations while remaining practical on the development machine (Intel i5-8250U, 8 GB RAM, GeForce 940MX 2 GB, Windows).

## 1. Current problems to fix

The current system already has V1 investigation, real public CAE corpus, LS-DYNA parsing, BM25, pgvector, RRF, structured CAE search and LangGraph Copilot. The remaining limitations are:

- hashing placeholder instead of real semantic embeddings;
- heuristic reranker instead of a real cross-encoder;
- 20-page PDF ingestion bound;
- no selective OCR;
- Docling not installed;
- no VLM visual-evidence fallback;
- LS-DYNA parsing bounded to main + direct includes;
- no full multimodal retrieval;
- no CISS/historical datasets yet;
- production hardening not yet implemented.

The current evaluation already shows why the embedding must be replaced: BM25-only is stronger than the current hashing-based dense leg, so **do not rewrite the retrieval architecture; replace the weak components.**

## 2. Target architecture

```text
PASSIVE SAFETY ENGINEER
        |
        v
Investigation UI
        |
        v
LangGraph Agent
        |
  +-----+--------------------+
  |                          |
  v                          v
Deterministic Tools      Retrieval
  |                          |
  |              +-----------+-----------+-----------+
  |              |           |           |           |
  |             BM25       Dense     Structured   Visual
  |                        pgvector     CAE SQL    evidence
  |              |           |           |           |
  |              +-----------+-----------+-----------+
  |                          |
  |                         RRF
  |                          |
  |                    Cross-Encoder
  |                      Reranker
  |                          |
  +--------------------------+
                             |
                    Authority/Grounding Gate
                             |
                          Evidence
                             |
                       AI Hypothesis
                             |
                      Engineer Review
```

## 3. Core strategy: Local deterministic + optional remote AI

The laptop must handle:

- source discovery and SHA-256;
- archive inspection;
- PDF routing;
- PyMuPDF extraction;
- LS-DYNA parsing and recursive include graphs;
- BM25;
- PostgreSQL/pgvector;
- structured CAE queries;
- Parquet processing;
- deterministic signal analysis;
- provenance and evaluation.

Heavy AI components must use replaceable providers:

- embeddings;
- reranker;
- VLM;
- LLM.

Never hard-code a cloud vendor or require a GPU.

## 4. Implementation order

Implement in this order:

```text
P0  Real embeddings
P1  Real cross-encoder reranker
P2  Remove PDF 20-page limit
P3  Recursive LS-DYNA include graph
P4  Selective OCR + PDF routing
P5  Docling complex-document adapter
P6  VLM visual-evidence fallback
P7  Multimodal retrieval
P8  CISS adapter when real data exists
P9  Historical-case adapter when authenticated data exists
P10 Production hardening
```

Do not implement P7-P10 before P0-P4 are stable.

## 5. P0 — Real semantic embeddings

Create an `EmbeddingProvider` interface. Benchmark real candidates on the actual passive-safety golden set rather than choosing by leaderboard alone.

At minimum benchmark:

```text
BGE-M3
one strong E5-family candidate
one current Qwen embedding-family candidate if available
```

Measure:

```text
Recall@5
Recall@10
MRR
latency/query
throughput
RAM
```

Record model/version/dimension/quantization/hardware/results in `evals/results/embedding_benchmark.json`.

The production path must not use the hashing placeholder after this phase. Keep it only as a test/mock provider if useful.

## 6. P1 — Real cross-encoder reranker

Replace the current heuristic reranker with:

```text
BM25 + Dense + Structured
        -> RRF
        -> top 30-50 candidates
        -> cross-encoder
        -> top 5-10
```

Create a `Reranker` interface with local, remote and mock implementations.

Benchmark at least two feasible candidates for:

```text
NDCG@10
MRR
Recall@10
latency
RAM
```

Reject a candidate that does not improve ranking quality on the domain golden set.

## 7. P2 — Remove the PDF 20-page limit

Replace the artificial limit with streaming/incremental page processing:

```text
PDF -> page iterator -> small batch -> extract -> validate -> persist -> release memory -> next page
```

Never load an entire large PDF into RAM.

Every page must have a state:

```text
DISCOVERED
EXTRACTED
OCR_REQUIRED
OCR_COMPLETE
VISUAL_REVIEW_REQUIRED
VALIDATED
FAILED
```

Every page remains traceable to `document_id`, page number and source SHA-256.

## 8. P3 — Recursive LS-DYNA include graph

Move from main + direct includes to recursive traversal:

```text
main.key
 ├── vehicle.k
 │    ├── material.k
 │    └── contact.k
 ├── dummy.key
 │    ├── thor.k
 │    └── head.k
 └── restraint.k
      └── belt.k
```

Track:

```text
RESOLVED
MISSING
CYCLE
DUPLICATE
AMBIGUOUS
OUTSIDE_ROOT
```

Never guess ambiguous files. Never execute deck content.

Create a manifest for every root deck containing all reachable files, hashes, statuses, keyword counts and parsed entity counts.

## 9. P4 — PDF routing and OCR

Use a router instead of one parser for every document:

```text
PDF
 |
 +-- simple digital -> PyMuPDF
 |
 +-- complex layout -> Docling
 |
 +-- scanned/image-only -> OCR
```

Simple documents should stay on the lightweight path. Complex tables/layouts should use Docling. OCR is selective, not universal.

OCR output must retain engine/version/confidence/page and low-confidence results become `NEEDS_REVIEW`.

## 10. P5 — Docling adapter

Install Docling only after the lightweight pipeline is stable. Add a `DocumentParser` abstraction with PyMuPDF and Docling implementations.

Docling is for:

- complex layout;
- reading order;
- complex tables;
- technical documents;
- structured sections.

If Docling is unavailable, fall back to PyMuPDF and mark quality honestly. Do not make the entire application dependent on Docling.

## 11. P6 — VLM visual evidence

VLM is a visual-evidence fallback, not the default PDF parser.

Use it when:

- table extraction is suspicious;
- chart structure matters;
- diagrams/schematics contain relationships;
- visual engineering information cannot be represented reliably by text extraction.

Pipeline:

```text
source page -> render -> VLM -> visual interpretation -> validation
```

Every VLM result must be labelled `VISUAL_INTERPRETATION`. Never convert an unvalidated VLM number into an authoritative engineering value.

## 12. P7 — Multimodal retrieval

After text retrieval is stable, support:

```text
Text    -> BM25 + Dense
Tables  -> Structured retrieval
Figures -> Visual retrieval
CAE     -> Structured SQL
```

Fuse at the evidence layer rather than pretending every modality is the same type of vector document.

## 13. Mandatory Structured + Hybrid RAG

The final text/CAE retrieval pipeline remains:

```text
QUERY
 |
 +-- BM25 / lexical search
 +-- Dense / pgvector
 +-- Structured CAE SQL
 |
 v
Reciprocal Rank Fusion
 |
v
Cross-encoder reranker
 |
v
Authority + metadata filtering
 |
v
Grounding gate
 |
v
Evidence
 |
v
LangGraph
```

**BM25 is mandatory.** Do not remove it. Exact identifiers such as `*MAT_024`, `*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE`, `*DATABASE_BINARY_D3PLOT`, `THOR-05F`, `HIC15`, `UN R94` and Part/Material IDs require lexical retrieval.

Structured search is mandatory for relationship questions such as:

```text
Which material does Part 1042 use?
Which contacts exist in this deck?
Which files are included?
Which database cards are present?
```

Do not ask the LLM to infer these relationships when the database already knows them.

## 14. Engineering relationship layer

Represent relationships in PostgreSQL first; do not add Neo4j merely for this purpose.

Examples:

```text
Deck -> includes -> File
Deck -> contains -> Part/Material/Contact/Control
Part -> uses -> Material/Section
Investigation -> compares -> Run
Investigation -> uses -> Evidence
Investigation -> produces -> Hypothesis
```

## 15. OKF role

Use OKF as a curated knowledge interchange/provenance layer, not as the primary database or vector store.

```text
raw source
 -> validated extraction
 -> OKF concept
 -> PostgreSQL
 -> retrieval
```

Preserve source, SHA-256, document, revision, page/line, authority and derivation information.

## 16. Provider architecture

All heavy AI components must support:

```text
local
remote
mock
```

Business logic depends only on interfaces:

```text
EmbeddingProvider
Reranker
VisionProvider
LLMProvider
```

This allows the same application to work on the laptop, with a free/remote API, or later in production without changing the investigation logic.

## 17. Offline degradation

The application must remain useful without an external AI provider.

Still available:

```text
BM25
structured CAE search
signal analysis
configuration diff
PDF extraction
LS-DYNA parsing
source viewer
```

If embeddings/reranker/VLM/LLM are unavailable, expose an explicit `UNAVAILABLE` state rather than silently substituting fake semantics.

## 18. No-silent-loss rules

PDF:

```text
original page count == accounted page count
```

except pages explicitly marked `FAILED` or `NEEDS_REVIEW`.

LS-DYNA:

```text
all reachable includes classified
all unknown keywords preserved
raw text preserved
source file preserved
line spans preserved
```

No unsupported engineering meaning may be invented.

## 19. Incremental ingestion

Every source is content-addressed.

If SHA-256 is unchanged:

```text
SKIP
```

If changed:

```text
REPROCESS
```

Do not duplicate large archives or keep unnecessary temporary page images.

## 20. Hardware rules

For the i5-8250U/8 GB/2 GB GPU machine:

- CPU-first;
- small batches;
- lazy loading;
- streaming;
- incremental indexing;
- remote AI where necessary;
- no full-corpus RAM loading;
- no requirement for GPU inference;
- no giant local model merely to claim SOTA.

## 21. Evaluation gates

No component is accepted because it is labelled SOTA.

Every candidate must pass:

```text
quality
latency
memory
storage
```

and domain retrieval metrics.

Required retrieval comparison:

```text
BM25
Dense
BM25 + Dense
BM25 + Dense + Structured
RRF
RRF + Reranker
```

Measure:

```text
Recall@5
Recall@10
MRR
NDCG@10
latency
```

## 22. PDF acceptance tests

Test:

```text
native PDF
scanned PDF
mixed PDF
table-heavy PDF
figure-heavy PDF
multi-column PDF
```

Verify page accounting, extraction, OCR fallback, tables, figures, warnings and failure visibility.

## 23. LS-DYNA acceptance tests

Test:

```text
simple.k
nested_include.key
unknown_keyword.k
malformed.k
comments.k
fixed_width.k
```

Verify keyword detection, raw preservation, line spans, recursive includes, missing files, cycles and ambiguity.

## 24. Real corpus acceptance test

Run the final pipeline against the actual `Knowledge source/` corpus, not only synthetic fixtures.

Generate:

```text
data/artifacts/level3_final_report.json
```

including:

```text
files discovered
archives
PDFs
K/KEY/INC files
pages
OCR pages
Docling pages
VLM pages
LS-DYNA keywords
resolved/missing/ambiguous includes
BM25 documents
embedding count
reranker status
retrieval metrics
failures
warnings
```

## 25. Commands

```powershell
uv sync
docker compose up -d postgres
uv run alembic upgrade head
uv run python scripts/profile_knowledge_sources.py
uv run python scripts/ingest_level3_pdfs.py
uv run python scripts/ingest_level3_cae_decks.py
uv run python scripts/generate_okf_concepts.py
uv run python scripts/index_knowledge.py
uv run pytest
uv run ruff check .
uv run mypy apps packages scripts tests conftest.py evals
```

Benchmarks:

```powershell
uv run python evals/embedding_benchmark.py
uv run python evals/reranker_benchmark.py
uv run python evals/level3_hybrid_eval.py
uv run python evals/scenario_eval.py
```

## 26. Failure policy

```text
Docling unavailable       -> PyMuPDF fallback
OCR unavailable           -> NEEDS_REVIEW
VLM unavailable           -> preserve page + visual-review status
Embedding unavailable     -> BM25 + structured remain available
Reranker unavailable      -> RRF remains available
LLM unavailable           -> deterministic investigation remains available
```

Never silently replace missing semantic capability with a fake embedding.

## 27. Explicit non-goals

Do not add merely for this milestone:

```text
Elasticsearch/OpenSearch just for BM25
Neo4j just for relationships
Kubernetes
huge local models
OCR on every page
VLM on every page
fake CISS data
fake historical investigations
automatic regulatory approval
autonomous engineering sign-off
```

## 28. Production hardening after Level 3.5

Only after ingestion/retrieval quality is proven:

```text
authentication
authorization
audit trail
investigation versioning
production DB
secret management
observability
rate limits
backup/restore
security testing
```

## 29. Claude Code execution rule

For each phase:

```text
inspect
plan
implement
test
report
```

Never perform a large uncontrolled rewrite.

At the end of every phase report:

```text
files changed
tests added/tests passed
retrieval metrics
performance
known limitations
next phase
```

Do not continue if a mandatory acceptance test fails.

## 30. Definition of done

```text
[ ] hashing embedding removed from production path
[ ] real embedding benchmark completed
[ ] real cross-encoder benchmark completed
[ ] 20-page PDF limit removed
[ ] page streaming implemented
[ ] recursive LS-DYNA include graph implemented
[ ] complete deck manifest implemented
[ ] PDF routing implemented
[ ] selective OCR implemented
[ ] Docling adapter implemented
[ ] VLM adapter implemented
[ ] structured CAE relationships exposed
[ ] BM25 retained and evaluated
[ ] dense retrieval retained and evaluated
[ ] RRF retained and evaluated
[ ] reranker retained and evaluated
[ ] grounding gate retained
[ ] provider abstraction implemented
[ ] offline degradation implemented
[ ] incremental ingestion implemented
[ ] real corpus evaluated
[ ] PDF completeness tests pass
[ ] LS-DYNA tests pass
[ ] retrieval benchmark passes
[ ] E2E investigation passes
[ ] README updated
[ ] ADRs updated
```

## 31. Final target

An engineer asking:

> **Why did chest deflection increase between Run A and Run B?**

should trigger:

```text
Run validation
 -> comparability
 -> configuration diff
 -> signal analysis
 -> first divergence
 -> structured CAE search
 -> BM25 + Dense
 -> RRF
 -> cross-encoder
 -> authority/grounding gate
 -> evidence
 -> LangGraph hypothesis
 -> engineer review
```

Every conclusion must remain traceable to:

```text
original source
page/line
CAE file/keyword
run
calculation
retrieval path
engineer decision
```

**This is the Level-3.5 target: a strong, evidence-backed engineering copilot without requiring a high-end local GPU.**
