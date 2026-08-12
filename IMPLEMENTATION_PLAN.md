# Passive Safety CAE Investigation Agent — Implementation Plan

**Version:** 2.0  
**Execution:** Claude Code + Claude Sonnet  
**Runtime LLM:** FreeLLMAPI abstraction  
**Development workflow skill:** Ponytail  
**Target hardware:** Intel i5-8250U / 8 GB RAM / NVIDIA 940MX 2 GB / Windows

---

# 1. Golden Rule

Do not start by building the agent.

Build:

```text
Engineering data
→ deterministic analysis
→ evidence
→ retrieval
→ agent
→ UI
→ production hardening
```

---

# 2. Phase 0 — Development Workflow Setup

## Goal

Prepare Claude Code and the repository before application code.

Tasks:

```text
1. Create Git repository.
2. Add all six design documents.
3. Obtain/review Ponytail.
4. Configure the relevant Ponytail skill/workflow.
5. Create project CLAUDE.md rules.
6. Create ADR directory.
7. Create development environment.
8. Create test framework.
```

Ponytail repository supplied by the user:

`https://github.com/DietrichGebert/ponytail`

FreeLLMAPI repository supplied by the user:

`https://github.com/tashfeenahmed/freellmapi`

Do not assume either repository is compatible until inspected.

---

# 3. Phase 1 — Repository Foundation

Create:

```text
passive-safety-cae-agent/
├── PRD.md
├── TRD.md
├── APP_FLOW.md
├── UI_UX_DESIGN_BRIEF.md
├── BACKEND_SCHEMA.md
├── IMPLEMENTATION_PLAN.md
│
├── docs/
│   └── ADR/
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
├── data/
├── evals/
├── scripts/
└── tests/
```

---

# 4. Phase 2 — Import Current Knowledge Corpus

The current documents available locally are:

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

Do not download more before creating the ingestion pipeline.

Create:

```text
knowledge/00_registry/source_manifest.yaml
```

Record:

```text
source_id
filename
category
authority
sha256
size
date_added
legal_status
processing_status
```

---

# 5. Phase 3 — Canonical Knowledge Structure

Create:

```text
knowledge/
├── 00_registry/
├── 01_regulations/
│   └── unece/
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

Do not delete the original downloaded copies.

---

# 6. Phase 4 — Document Extraction

Pipeline:

```text
Register
→ SHA256
→ Inspect
→ PyMuPDF extraction
→ Quality gate
→ Section detection
→ Table extraction
→ Figure extraction
→ Equation extraction
→ Markdown
→ Chunking
→ Metadata
```

First test on:

```text
UN_R94
```

Then:

```text
LS-DYNA Theory R17
```

Then:

```text
LS-DYNA Keyword Vol I R17
```

Do not process all documents simultaneously on the 8 GB machine.

---

# 7. Phase 5 — Synthetic CAE Dataset

Create ten scenarios:

```text
SCN-001 Belt revision
SCN-002 Pretensioner timing
SCN-003 Airbag timing
SCN-004 Crash pulse change
SCN-005 Seat position
SCN-006 Dummy positioning
SCN-007 Contact/friction
SCN-008 Signal processing
SCN-009 Model revision
SCN-010 Numerical-quality failure
```

Generate:

```text
Run A
Run B
configuration
signals
ground truth
```

Use deterministic seeds.

---

# 8. Phase 6 — Domain Model

Implement:

```text
Project
Vehicle
ModelVersion
Component
ComponentRevision
SimulationRun
Artifact
SignalDefinition
Signal
```

Create PostgreSQL migrations.

Add tests.

---

# 9. Phase 7 — Deterministic Analysis

Implement before LangGraph:

```text
run_quality_gate()
assess_comparability()
compare_global_response()
compare_configuration()
calculate_signal_features()
detect_first_divergence()
```

No LLM calls.

---

# 10. Phase 8 — Quality Gate

Implement adapters:

```text
QualityProvider
├── SyntheticProvider
└── LS-DYNAProvider
```

Initial LS-DYNA provider may operate on normalized synthetic/result fixtures rather than requiring commercial solver execution.

---

# 11. Phase 9 — Configuration Diff

Implement:

```text
component hierarchy
parameter diff
classification
provenance
```

Test against SCN-001 through SCN-010.

---

# 12. Phase 10 — Signal Engine

Implement:

```text
peak
time_to_peak
rise_time
duration
integral
correlation
phase shift
first divergence
```

Each result must contain:

```text
algorithm
version
parameters
filter
alignment
source signal
source run
```

---

# 13. Phase 11 — Knowledge RAG

Start with only:

```text
UN R94
LS-DYNA Theory R17
LS-DYNA Keyword Vol I R17
```

Implement:

```text
PostgreSQL FTS
+
pgvector
+
metadata filtering
+
RRF
```

Then expand to:

```text
UN R16
UN R95
UN R129
LS-DYNA Vol II
LS-DYNA Vol III
LS-DYNA User Guide
Examples
PAM specification
```

---

# 14. Phase 12 — RAG Evaluation

Create a golden dataset.

Metrics:

```text
Recall@5
Recall@10
MRR
citation accuracy
source authority correctness
revision correctness
```

For every answer verify:

```text
Did the system retrieve the correct document?
Did it retrieve the correct section/page?
Did it cite the correct source?
Did it avoid unsupported claims?
```

---

# 15. Phase 13 — Historical Cases

Use synthetic historical cases first.

Structure:

```text
case_id
question
runs
changed_factor
signals
features
hypotheses
finding
evidence
```

Retrieval:

```text
metadata
+
semantic
+
signal feature similarity
```

---

# 16. Phase 14 — FreeLLMAPI Integration

Create:

```text
LLMProvider
```

Implement:

```text
FreeLLMAPIProvider
MockProvider
```

Configuration:

```text
LLM_BASE_URL
LLM_MODEL
LLM_API_KEY
```

Do not hardcode the model.

FreeLLMAPI availability, provider quotas, rate limits and model selection must be treated as runtime configuration.

---

# 17. Phase 15 — LangGraph Agent

Graph:

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

Conditional branches:

```text
request_more_signal
missing_evidence
reject_hypothesis
comparison_blocked
```

---

# 18. Phase 16 — Evidence Layer

Implement:

```text
Evidence
Claim
Hypothesis
Finding
Decision
```

Required trace:

```text
Claim
→ Evidence
→ Signal/Document
→ Artifact
→ Hash
```

---

# 19. Phase 17 — UI

Build in this order:

```text
Dashboard
Run Browser
Run Detail
New Investigation
Run Identity
Quality
Comparability
Global Response
Configuration Diff
Signal Workspace
Mechanism Review
Evidence
Hypotheses
Knowledge
Review
Report
```

---

# 20. Phase 18 — Full Investigation Workspace

Combine:

```text
Run A
Run B
charts
configuration
events
evidence
hypotheses
AI assistant
engineer review
```

The UI must work even if the LLM is unavailable.

---

# 21. Phase 19 — Evaluation Harness

Evaluate:

### Numerical

```text
feature accuracy
divergence accuracy
configuration diff accuracy
```

### Retrieval

```text
Recall@K
MRR
citation accuracy
```

### Agent

```text
tool selection
evidence grounding
hypothesis quality
uncertainty handling
```

### Product

```text
investigation completion
reviewer corrections
time to finding
```

---

# 22. Phase 20 — Production Hardening

Implement:

```text
authentication
authorization
audit logs
file validation
job management
retry policies
observability
backup
secrets management
configuration
```

---

# 23. Phase 21 — Later Solver Integrations

Future:

```text
LS-DYNA
PAM-CRASH
ANSYS
```

Architecture:

```text
SolverAdapter
├── SyntheticAdapter
├── LSDYNAAdapter
├── PAMCrashAdapter
└── AnsysAdapter
```

V1 must not require direct commercial solver execution.

---

# 24. Phase Gates

## Gate 1

Two runs can be registered and queried.

## Gate 2

Synthetic quality/configuration/signal analysis works.

## Gate 3

Knowledge ingestion works with page-level citations.

## Gate 4

RAG passes retrieval benchmark.

## Gate 5

Historical retrieval works.

## Gate 6

Agent completes synthetic investigation.

## Gate 7

Engineer can complete investigation in UI.

## Gate 8

Reproducibility and audit pass.

---

# 25. Definition of Done

A feature is complete only when:

```text
implementation
+
tests
+
error handling
+
logging
+
documentation
+
evaluation
+
provenance
```

---

# 26. Claude Code Execution Protocol

Before every phase:

```text
1. Read relevant design documents.
2. Inspect repository.
3. State implementation plan.
4. Identify dependencies.
5. Implement.
6. Run tests.
7. Run evaluation.
8. Inspect failures.
9. Update documentation.
10. Report completion.
```

Claude must not silently move to the next phase if a gate fails.

---

# 27. Do Not Build These Yet

```text
No Kubernetes
No Kafka
No Elasticsearch
No Qdrant
No giant local model
No multi-agent swarm
No automatic commercial solver execution
No automatic engineering approval
No full ANSYS corpus
No full PAM-CRASH corpus
```

---

# 28. First Coding Task

The first coding task after the documents are finalized is NOT the agent.

It is:

```text
Repository bootstrap
+
Ponytail workflow setup
+
PostgreSQL
+
FastAPI
+
Next.js
+
Alembic
+
test framework
+
source registry
```

Then stop and validate the foundation.
