# Product Requirements Document — Safety Assistant v2

## Product

**Safety Assistant — Passive Safety Regulatory Intelligence Workspace**

## Vision

Give passive-safety engineers a trusted workspace to search, understand and verify regulatory requirements across approved UNECE/passive-safety documents and their own authorized project documents, while maintaining traceable evidence and version provenance.

## Primary persona — Passive Safety Engineer

### Goals
- find applicable regulatory clauses quickly;
- confirm exact numerical requirements;
- compare requirements across versions/documents;
- inspect annexes/tables;
- ask follow-up questions without losing context;
- add a new regulation or internal document;
- know when the uploaded document is safely searchable;
- return to past investigations;
- cite the source in engineering work.

### Needs
- correctness before fluency;
- exact evidence;
- version traceability;
- fast search;
- clear upload/processing states;
- privacy for project documents.

## Secondary personas

### Knowledge Admin
Can approve/promote authoritative documents, inspect failed ingestion, reprocess and monitor corpus freshness/completeness.

### Auditor / Reviewer
Can inspect answer/citation history and provenance according to role, without changing authoritative data.

## Jobs to be done

### JTBD-01 — Regulatory question
When I need a requirement, I want to ask naturally and see the exact source, so I can use the result without manually scanning a long PDF.

### JTBD-02 — Evidence verification
When an answer contains a technical value, I want the clause/page/version, so I can verify it before an engineering decision.

### JTBD-03 — New document
When I receive a new regulation or project PDF, I want to upload it and know when it is safely indexed.

### JTBD-04 — Continue investigation
When I return later, I want to resume the same thread and source scope.

### JTBD-05 — Private project material
When I upload confidential project material, I want it isolated to authorized users/workspaces.

## MVP scope

### Authentication
- login/logout;
- production OIDC-ready model;
- safe development auth mode;
- user profile;
- role/membership.

### Workspace
- active workspace;
- source-scope indicator;
- recent conversations;
- recent documents.

### Chat
- new chat;
- persistent threads;
- streaming where supported;
- citations;
- evidence panel;
- explicit insufficient-evidence mode;
- source filters;
- rename/archive.

### Documents
- drag/drop PDF upload;
- document list/detail;
- status;
- source scope;
- version metadata;
- validation errors;
- ingestion progress;
- archive/delete subject to policy.

### Ingestion
- asynchronous;
- deterministic lifecycle;
- validation;
- parsing;
- structure-aware chunking;
- embeddings;
- index;
- QA;
- idempotency/deduplication;
- quarantine.

### User history/memory
- durable conversation history;
- explicit preferences;
- default workspace;
- no hidden regulatory memory.

### Administration
- organization corpus;
- failed jobs;
- authoritative promotion/approval;
- role-gated audit visibility.

## Out of scope for MVP

- unrestricted web browsing;
- fully autonomous regulatory research agents;
- automatic authoritative promotion of uploads;
- graph database without measured need;
- fine-tuning;
- real-time collaborative document editing;
- mobile-first admin workflows;
- replacing source PDFs with model-generated summaries.

## Functional requirements

### Auth
- **FR-AUTH-01**: every authenticated user has a stable `user_id`.
- **FR-AUTH-02**: conversation/document queries are backend authorization-scoped.

### Chat
- **FR-CHAT-01**: create conversation.
- **FR-CHAT-02**: messages survive browser restart and re-login.
- **FR-CHAT-03**: factual answers persist citations/evidence references.
- **FR-CHAT-04**: user can choose authorized source scope.

Source choices:

```text
Verified regulations
Workspace documents
My private documents
All authorized sources
```

### Documents
- **FR-DOC-01**: upload PDF.
- **FR-DOC-02**: upload returns document/version/job identifiers.
- **FR-DOC-03**: processing status is visible.
- **FR-DOC-04**: only READY versions participate in retrieval.
- **FR-DOC-05**: failure includes a user-safe reason and diagnostic reference.
- **FR-DOC-06**: duplicate/version behavior is deterministic.

### RAG
- **FR-RAG-01**: authorization and temporal/source filters happen before ranking.
- **FR-RAG-02**: answers contain evidence references.
- **FR-RAG-03**: unsupported generated numeric/identifier claims are rejected or removed deterministically.
- **FR-RAG-04**: provider failure can degrade to evidence-only.

### Memory
- **FR-MEM-01**: persistent memory is limited to conversation continuity and explicit preferences.
- **FR-MEM-02**: memory cannot be cited as regulatory evidence.

### Admin
- **FR-ADMIN-01**: only privileged roles can promote a document to authoritative organization scope.

## Non-functional requirements

### Correctness
- retrieval regression gates;
- citation contract;
- no production fake embeddings/providers;
- stale-version exclusion.

### Security
- least privilege;
- user/workspace/org scoping;
- secure sessions/tokens;
- upload bounds;
- SSRF-safe acquisition;
- prompt-injection tests;
- secret management.

### Accessibility
Core workflows target WCAG 2.2 AA behavior.

### Auditability
Persist enough IDs/config/version data to trace an answer to exact document versions/chunks.

## Success metrics

### User outcome
- median time to verified requirement;
- evidence-open rate;
- task completion;
- usefulness rating.

### Retrieval
- Recall@5/10/20;
- MRR;
- nDCG@10;
- clause hit rate.

### Generation
- citation precision/completeness;
- unsupported-claim rate;
- refusal accuracy;
- numeric accuracy.

### Product
- ingestion success rate;
- time to READY;
- conversation restore success;
- document search adoption.

## Acceptance scenarios

### A — first question
Login → new chat → ask a regulation question → grounded answer → open citation → verify version/clause/page.

### B — private upload
Upload private PDF → processing → READY → ask scoped question → answer cites upload → another user cannot access it.

### C — bad document
Invalid file → fails validation/quarantine → UI explains failure → retrieval cannot access it.

### D — resume
Create conversation → logout → login → history restores messages/citations/source scope.
