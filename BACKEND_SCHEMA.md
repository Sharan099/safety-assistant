# Passive Safety CAE Investigation Agent — Backend Schema

**Version:** 2.0  
**Database:** PostgreSQL + pgvector  
**Large numerical data:** Parquet/HDF5  
**Analytics:** DuckDB

---

# 1. Schema Philosophy

The schema is designed around the engineering investigation, not around the UI.

Core chain:

```text
Run
→ Analysis
→ Evidence
→ Claim
→ Hypothesis
→ Finding
→ Decision
```

Never collapse these into one AI response.

---

# 2. Core Entities

```text
Organization
User
Project
Vehicle
ModelVersion
Component
ComponentRevision
SimulationRun
Artifact
SignalDefinition
Signal
SignalFeature
```

---

# 3. Knowledge Entities

```text
KnowledgeSource
Document
DocumentRevision
SourceSnapshot
DocumentPage
DocumentSection
DocumentChunk
DocumentTable
DocumentFigure
DocumentEquation
Embedding
```

---

# 4. Investigation Entities

```text
Investigation
InvestigationRun
InvestigationMetric
QualityGateResult
ComparabilityAssessment
ConfigurationDiff
SignalAnalysis
AnalysisEvent
Hypothesis
Evidence
Claim
HypothesisEvidenceLink
EvidenceContradiction
Finding
RecommendedAction
ControlledComparisonRequest
EngineerReview
```

---

# 5. Provenance Entities

```text
ProcessingJob
ProcessingStep
SignalTransformation
ArtifactHash
```

---

# 6. Organizations

```text
organizations
id UUID PK
name TEXT
created_at TIMESTAMP
updated_at TIMESTAMP
```

---

# 7. Users

```text
users
id UUID PK
organization_id UUID FK
email TEXT UNIQUE
display_name TEXT
role TEXT
created_at TIMESTAMP
updated_at TIMESTAMP
```

Roles:

```text
ENGINEER
REVIEWER
ADMIN
```

---

# 8. Projects

```text
projects
id UUID PK
organization_id UUID FK
name TEXT
description TEXT
status TEXT
created_at TIMESTAMP
updated_at TIMESTAMP
```

---

# 9. Vehicles

```text
vehicles
id UUID PK
project_id UUID FK
name TEXT
programme TEXT
vehicle_type TEXT
metadata JSONB
created_at TIMESTAMP
```

---

# 10. Model Versions

```text
model_versions
id UUID PK
vehicle_id UUID FK
version TEXT
parent_version_id UUID NULL FK
source_artifact_id UUID NULL FK
metadata JSONB
created_at TIMESTAMP
```

---

# 11. Components

```text
components
id UUID PK
project_id UUID FK
name TEXT
component_type TEXT
parent_component_id UUID NULL FK
metadata JSONB
```

Types:

```text
BODY
SEAT
BELT
AIRBAG
DUMMY
MATERIAL
CONTACT
CONNECTOR
STRUCTURE
SOLVER_CONTROL
OTHER
```

---

# 12. Component Revisions

```text
component_revisions
id UUID PK
component_id UUID FK
revision TEXT
source_artifact_id UUID NULL FK
parameters JSONB
created_at TIMESTAMP
```

---

# 13. Simulation Runs

```text
simulation_runs
id UUID PK
project_id UUID FK
vehicle_id UUID FK
model_version_id UUID FK

run_id TEXT UNIQUE
parent_run_id UUID NULL FK

solver TEXT
solver_version TEXT

dummy_version TEXT
impact_type TEXT
impact_speed DOUBLE PRECISION
barrier TEXT

seat_configuration JSONB
restraint_configuration JSONB

result_processing_version TEXT

status TEXT
quality_status TEXT

started_at TIMESTAMP
completed_at TIMESTAMP

metadata JSONB
created_at TIMESTAMP
```

---

# 14. Artifacts

```text
artifacts
id UUID PK
simulation_run_id UUID NULL FK

artifact_type TEXT
filename TEXT
storage_uri TEXT

mime_type TEXT
size_bytes BIGINT
sha256 TEXT UNIQUE

source_type TEXT
created_at TIMESTAMP
```

---

# 15. Signal Definitions

```text
signal_definitions
id UUID PK
name TEXT
canonical_name TEXT
domain TEXT
unit TEXT
quantity TEXT
source_standard TEXT NULL
metadata JSONB
created_at TIMESTAMP
```

---

# 16. Signals

```text
signals
id UUID PK
simulation_run_id UUID FK
signal_definition_id UUID FK

name TEXT
source_channel TEXT
storage_uri TEXT

sampling_rate DOUBLE PRECISION
start_time DOUBLE PRECISION
end_time DOUBLE PRECISION
unit TEXT

processing_version_id UUID NULL FK
metadata JSONB
created_at TIMESTAMP
```

Raw samples remain in Parquet/HDF5.

---

# 17. Signal Features

```text
signal_features
id UUID PK
signal_id UUID FK

feature_name TEXT
value DOUBLE PRECISION
unit TEXT

algorithm_version TEXT
parameters JSONB
created_at TIMESTAMP
```

---

# 18. Knowledge Sources

```text
knowledge_sources
id UUID PK
source_key TEXT UNIQUE
source_type TEXT
category TEXT
authority_level TEXT
publisher TEXT
source_url TEXT NULL
local_path TEXT
metadata JSONB
created_at TIMESTAMP
```

Types:

```text
AUTHORITATIVE
OFFICIAL_DOCUMENTATION
INTERNAL_APPROVED
HISTORICAL
REFERENCE
SYNTHETIC
```

---

# 19. Documents

```text
documents
id UUID PK
knowledge_source_id UUID FK

document_key TEXT UNIQUE
title TEXT
document_type TEXT
publisher TEXT
language TEXT
created_at TIMESTAMP
```

---

# 20. Document Revisions

```text
document_revisions
id UUID PK
document_id UUID FK

revision_label TEXT
effective_date DATE NULL
source_snapshot_id UUID FK

extractor TEXT
extractor_version TEXT
status TEXT

created_at TIMESTAMP
```

Revisions are immutable.

---

# 21. Source Snapshots

```text
source_snapshots
id UUID PK

storage_uri TEXT
filename TEXT
sha256 TEXT UNIQUE
size_bytes BIGINT

retrieved_at TIMESTAMP
source_url TEXT NULL

metadata JSONB
```

---

# 22. Pages

```text
document_pages
id UUID PK
document_revision_id UUID FK

page_number INTEGER
text_content TEXT
markdown_content TEXT

page_image_uri TEXT NULL

text_quality DOUBLE PRECISION
layout_quality DOUBLE PRECISION
ocr_used BOOLEAN
ocr_confidence DOUBLE PRECISION NULL

metadata JSONB
```

---

# 23. Sections

```text
document_sections
id UUID PK
document_revision_id UUID FK
parent_section_id UUID NULL FK

title TEXT
section_number TEXT
start_page INTEGER
end_page INTEGER
content TEXT
```

---

# 24. Chunks

```text
document_chunks
id UUID PK
document_revision_id UUID FK
section_id UUID NULL FK
page_id UUID NULL FK
parent_chunk_id UUID NULL FK

chunk_type TEXT
content TEXT
token_count INTEGER

source_locator JSONB
created_at TIMESTAMP
```

---

# 25. Embeddings

```text
embeddings
id UUID PK
chunk_id UUID NULL FK

model_name TEXT
model_version TEXT
dimensions INTEGER

embedding VECTOR

created_at TIMESTAMP
```

---

# 26. Tables

```text
document_tables
id UUID PK
document_revision_id UUID FK
page_id UUID FK

table_number TEXT
markdown_uri TEXT
json_uri TEXT
image_uri TEXT NULL

bounding_box JSONB
extraction_method TEXT
quality_score DOUBLE PRECISION
created_at TIMESTAMP
```

---

# 27. Figures

```text
document_figures
id UUID PK
document_revision_id UUID FK
page_id UUID FK

figure_number TEXT
caption TEXT
image_uri TEXT
bounding_box JSONB
figure_type TEXT
extraction_method TEXT
metadata JSONB
```

---

# 28. Equations

```text
document_equations
id UUID PK
document_revision_id UUID FK
page_id UUID FK

equation_number TEXT
latex TEXT NULL
image_uri TEXT NULL
bounding_box JSONB

extraction_method TEXT
confidence DOUBLE PRECISION
```

---

# 29. Investigations

```text
investigations
id UUID PK
project_id UUID FK
created_by UUID FK

title TEXT
question TEXT
primary_metric_id UUID NULL

state TEXT
decision TEXT NULL

created_at TIMESTAMP
updated_at TIMESTAMP
closed_at TIMESTAMP NULL
```

---

# 30. Investigation Runs

```text
investigation_runs
id UUID PK
investigation_id UUID FK
simulation_run_id UUID FK
role TEXT
```

Roles:

```text
BASELINE
COMPARISON
PHYSICAL_TEST_REFERENCE
```

---

# 31. Investigation Metrics

```text
investigation_metrics
id UUID PK
investigation_id UUID FK
signal_definition_id UUID FK

priority INTEGER
is_primary BOOLEAN
reason TEXT
```

---

# 32. Quality Results

```text
quality_gate_results
id UUID PK
investigation_id UUID FK
simulation_run_id UUID FK

check_type TEXT
status TEXT
value JSONB
threshold JSONB
explanation TEXT
evidence_id UUID NULL FK

created_at TIMESTAMP
```

---

# 33. Comparability

```text
comparability_assessments
id UUID PK
investigation_id UUID FK

dimension TEXT
status TEXT
explanation TEXT
evidence_ids UUID[]

created_at TIMESTAMP
```

---

# 34. Configuration Diff

```text
configuration_diffs
id UUID PK
investigation_id UUID FK
component_id UUID NULL FK

path TEXT

run_a_value JSONB
run_b_value JSONB

change_status TEXT
change_classification TEXT

created_at TIMESTAMP
```

---

# 35. Signal Analysis

```text
signal_analyses
id UUID PK
investigation_id UUID FK
signal_definition_id UUID FK

run_a_signal_id UUID FK
run_b_signal_id UUID FK

alignment_method TEXT
filtering_method TEXT
metrics JSONB

algorithm_version TEXT
parameters JSONB

created_at TIMESTAMP
```

---

# 36. Analysis Events

```text
analysis_events
id UUID PK
signal_analysis_id UUID FK

event_type TEXT
time_ms DOUBLE PRECISION

algorithm_version TEXT
threshold JSONB
window JSONB
alignment_method TEXT

source_signal_id UUID FK

metadata JSONB
created_at TIMESTAMP
```

---

# 37. Hypotheses

```text
hypotheses
id UUID PK
investigation_id UUID FK

title TEXT
description TEXT
status TEXT
confidence_basis TEXT

affected_components JSONB
missing_evidence JSONB
recommended_isolation JSONB

created_by TEXT
created_at TIMESTAMP
updated_at TIMESTAMP
```

---

# 38. Evidence

```text
evidence
id UUID PK
investigation_id UUID FK

evidence_type TEXT
source_type TEXT
source_id UUID NULL
source_locator JSONB

content TEXT
value JSONB

calculation_version TEXT
source_hash TEXT

created_at TIMESTAMP
```

Types:

```text
OBSERVED
CALCULATED
DOCUMENTARY
HISTORICAL
INFERRED
```

---

# 39. Claims

```text
claims
id UUID PK
investigation_id UUID FK

claim TEXT
status TEXT
created_by TEXT
created_at TIMESTAMP
```

---

# 40. Hypothesis Evidence Links

```text
hypothesis_evidence_links
id UUID PK
hypothesis_id UUID FK
evidence_id UUID FK

relationship TEXT
weight DOUBLE PRECISION NULL
notes TEXT
```

Relationships:

```text
SUPPORTS
CONTRADICTS
CONTEXT
```

---

# 41. Contradictions

```text
evidence_contradictions
id UUID PK
investigation_id UUID FK

evidence_a_id UUID FK
evidence_b_id UUID FK

description TEXT
created_at TIMESTAMP
```

---

# 42. Findings

```text
findings
id UUID PK
investigation_id UUID FK

finding TEXT
status TEXT

reviewed_by UUID NULL FK
reviewed_at TIMESTAMP NULL

created_at TIMESTAMP
```

---

# 43. Recommended Actions

```text
recommended_actions
id UUID PK
investigation_id UUID FK

action_type TEXT
description TEXT
priority TEXT
status TEXT

created_at TIMESTAMP
```

---

# 44. Controlled Comparison Request

```text
controlled_comparison_requests
id UUID PK
investigation_id UUID FK
hypothesis_id UUID FK

requested_change JSONB
controlled_variables JSONB

reason TEXT
status TEXT
created_at TIMESTAMP
```

V1 records requests; it does not autonomously launch commercial solver jobs.

---

# 45. Engineer Review

```text
engineer_reviews
id UUID PK
investigation_id UUID FK
reviewer_id UUID FK

decision TEXT
comment TEXT
created_at TIMESTAMP
```

---

# 46. Processing Jobs

```text
processing_jobs
id UUID PK
job_type TEXT
status TEXT

source_artifact_id UUID NULL FK

started_at TIMESTAMP
completed_at TIMESTAMP NULL
error TEXT NULL

metadata JSONB
```

---

# 47. Processing Steps

```text
processing_steps
id UUID PK
processing_job_id UUID FK

step_name TEXT
software TEXT
software_version TEXT

parameters JSONB
input_hashes TEXT[]
output_hashes TEXT[]

status TEXT
started_at TIMESTAMP
completed_at TIMESTAMP
```

---

# 48. Signal Transformations

```text
signal_transformations
id UUID PK
signal_id UUID FK

transformation_type TEXT
algorithm_version TEXT
parameters JSONB

input_signal_id UUID NULL FK
created_at TIMESTAMP
```

---

# 49. Indexes

Create relational indexes for:

```text
project_id
run_id
model_version_id
simulation_run_id
signal_definition_id
document_revision_id
section_id
investigation_id
state
evidence investigation
hypothesis investigation
quality investigation
comparability investigation
```

Use PostgreSQL FTS for:

```text
document chunks
hypotheses
claims
historical case narrative
```

Use pgvector for:

```text
document chunks
historical case embeddings
```

---

# 50. Data Invariants

1. Every source artifact has a hash.
2. Every document chunk belongs to a revision.
3. Every calculated value has a processing version.
4. Every investigation references exact runs.
5. Every important claim can link to evidence.
6. Supporting and contradicting evidence are separate relationships.
7. Synthetic data is explicitly labelled.
8. Unknown is distinct from pass.
9. Source authority is preserved.
10. Engineer decisions are auditable.
