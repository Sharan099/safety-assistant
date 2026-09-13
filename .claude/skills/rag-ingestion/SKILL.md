---
name: rag-ingestion
description: Use for document upload, validation, parsing, chunking, embedding, versioning, indexing, ingestion jobs, and READY activation changes.
---

# RAG Ingestion

Read `docs/product/02_TRD.md` when needed.

Required properties:
- idempotent;
- checksum-aware;
- versioned;
- safe validation boundary;
- async;
- bounded retries;
- failed/quarantined versions excluded from retrieval;
- production fakes forbidden;
- atomic activation.

For every change:
1. map current stage;
2. define state transition;
3. define retry/idempotency;
4. define authorization/scope;
5. add success/failure tests;
6. check reprocessing/version behavior;
7. verify serving-index consistency.
