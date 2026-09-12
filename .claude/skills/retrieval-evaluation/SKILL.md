---
name: retrieval-evaluation
description: Use whenever retrieval, embeddings, sparse search, RRF, reranking, expansion, filters, chunking, or indexing behavior changes.
---

# Retrieval Evaluation

Before:
- locate baseline artifact;
- identify affected query categories;
- record current metrics.

Rules:
- no benchmark hard-coding;
- evaluate retrieval separately from generation;
- preserve authorization filters;
- preserve stale-version exclusion.

Measure as applicable:
- Recall@5/10/20;
- MRR;
- nDCG@10;
- hit rate;
- latency.

After:
- compare baseline vs candidate;
- identify category regressions;
- block if required floor fails;
- record git/config/dataset identifiers.
