# Regulation Knowledge-Base & Registry Pipeline — PRD (Final)

**Version:** 1.0
**Status:** Draft for engineering / agent build
**System:** Automotive Safety RAG & Regulation Discovery System
**Scope:** The data pipeline only — acquisition → validation → storage → registry →
parsing → chunking → embedding → indexing → retrieval. Not the chat UI or LLM
answer-generation layer.

---

## 1. Summary

The chat assistant answers from chunks in `safety_registry.db` (today: 54
regulations, 57 documents, 7,650 chunks). That database is the knowledge base.
This PRD defines a reliable, automatic pipeline that keeps the knowledge base
correct and current: regulations are fetched automatically from official
sources, every file is validated before it is allowed in, everything lives in
one place, a tracker decides what is genuinely new, and only changed documents
are parsed, chunked, embedded, and indexed. The registry tracks versions,
revisions, and metadata as the system of record.

The pipeline is built **on top of the existing code** (`scheduler/`, `crawler/`,
`registry/`, `parser/`, `vectorization/`). Each step is a gate: a document
cannot reach the next step until it passes the current one.

---

## 2. Current state (baseline, from audit)

- Chat reads chunks from `safety_registry.db` via `registry/search.py`
  (hybrid dense + keyword, then rerank, then LLM).
- A scheduler exists (`scheduler/jobs.py`) but runs `mock=True` weekly and needs
  a Celery worker — so **no real automatic downloads happen today**.
- Real downloads only via manual `python scripts/run_live_crawler.py`, which
  writes to `data/downloads/` — **a different place** from `storage/`.
- Files are **scattered** across `storage/<Authority>/` and `data/downloads/`.
- Version control (`registry/version_control.py`) compares **file** SHA-256 and
  can mark older copies `SUPERSEDED`.
- Chunking is **flat ~1,000-char windows** (200 overlap) with section-label
  detection — not clause/section-structured. `hierarchical_chunker.py` is not in
  this commit.
- Coverage is a **starter set**, not complete; some files may be
  synthetic/placeholder.

---

## 3. Problems this PRD fixes

1. No real automatic refresh (scheduler is mock-only, needs a worker).
2. Files scattered across two locations.
3. No reliable validation that a downloaded file is a real, official, non-synthetic PDF.
4. Dedup is on raw file bytes (fragile — PDF metadata changes flip the hash).
5. No tracker that gates embedding to only changed documents.
6. Flat chunking loses legal structure.
7. No completeness signal for coverage.

---

## 4. Goals and non-goals

**Goals**
- Fully automatic, periodic acquisition from official sources — no manual push.
- Validate every PDF before it enters the knowledge base.
- One canonical storage location.
- A change tracker that gates the embedding pipeline (incremental).
- A robust registry: versions, revisions, rich metadata, supersede logic.
- Structure-aware chunking with rich per-chunk metadata.
- Incremental embedding/indexing — only changed documents/chunks.
- Hybrid retrieval (BM25 + dense) with metadata filtering + cross-encoder rerank.
- Unit tests on every step; an integration test that proves the flow order.

**Non-goals (now)**
- Changing the chat UI or answer-generation logic.
- Vector compression/quantization (e.g. the "TurboVec" approach) — **deferred**
  until the collection reaches millions of chunks. At 7,650 chunks it adds
  complexity for no benefit.
- Guaranteeing 100% regulatory completeness automatically — the system reports
  coverage and gaps; a human owns sign-off.

---

## 5. Functional requirements

### 5.1 Acquisition (automatic, periodic)
- **FR-1** A scheduled job runs automatically on a configurable interval
  (e.g. quarterly heavy crawl) with **no manual step**. `mock` mode is off in
  production and clearly separated from real runs.
- **FR-2** A lighter, more frequent job (e.g. weekly) does a cheap change check
  (HTTP `HEAD` / `ETag` / `Last-Modified`) and only triggers a full fetch for
  documents that changed.
- **FR-3** Sources are a fixed allow-list of official domains per authority
  (UNECE, NHTSA/regulations.gov, Euro NCAP, IIHS, EU, China). The crawler refuses
  anything outside the list. All fetches are HTTPS with certificate verification.
- **FR-4** The scheduler's runtime dependency (Celery worker, etc.) is documented
  and health-checked; if the worker is down, this is visibly reported, not silent.

### 5.2 Validation (before entering the knowledge base)
- **FR-5** Downloaded files land first in a **staging/quarantine** area, never
  directly in the canonical store.
- **FR-6** A file is accepted only if it passes, in order:
  (a) magic-bytes check (`%PDF-`); (b) opens with the PDF library and has > 0
  pages; (c) size within an expected min/max; (d) the expected regulation
  identifier (e.g. "UN R94") is found in extracted text.
- **FR-7** Synthetic/placeholder detection: files matching known placeholder
  patterns or with no real text layer are rejected (image-only scans are flagged
  for OCR, not silently dropped).
- **FR-8** Rejected files stay in quarantine with a logged reason; they never
  reach storage or the registry.

### 5.3 Single canonical storage
- **FR-9** There is exactly **one** canonical location for accepted PDFs:
  `storage/<Authority>/<filename>`. `data/downloads/` is demoted to transient
  staging only (or removed). No accepted document lives in two places.
- **FR-10** A documented, deterministic naming/path convention ties each stored
  file to its authority and regulation.

### 5.4 Change tracker (gates embedding)
- **FR-11** A tracker/manifest records every known document and its current
  content hash, version, and ingest status.
- **FR-12** Before parsing/embedding, the pipeline consults the tracker and
  processes **only** documents marked new or changed. Unchanged documents are skipped.
- **FR-13** Dedup uses a hash of **normalized extracted text**, not raw PDF
  bytes. Same hash → duplicate (skip). Different hash → revision. No prior record
  → new document.

### 5.5 Registry (system of record)
- **FR-14** The registry stores, per document: authority, regulation id,
  amendment/series, version number, content hash, source URL, fetch timestamp,
  file path, and status (active / superseded / quarantined).
- **FR-15** On a new revision, the prior version is **kept and marked
  superseded**, not overwritten — older series remain retrievable for legacy programs.
- **FR-16** Amendment/series is parsed from document front matter where possible,
  not only from filename, and discrepancies are flagged.
- **FR-17** The registry can report coverage: documents per authority, and a
  configurable expected-set so gaps are visible.

### 5.6 Parsing
- **FR-18** Parsing extracts and preserves structure: headings, sections/clauses,
  tables, references, and annexes — not just flat page text.
- **FR-19** Tables are extracted as whole units (never split mid-table downstream).

### 5.7 Structure-aware chunking
- **FR-20** Chunks are formed on **section/clause/annex boundaries**, not fixed
  character windows. Oversized sections are split with overlap; very long single
  clauses are kept whole.
- **FR-21** Every chunk carries rich metadata: authority, regulation id,
  amendment/series, document, page, section/clause id, heading path, chunk type
  (section / paragraph / table), and content hash.
- **FR-22** Parent–child relationships are preserved so retrieval can expand a
  precise child chunk to its enclosing section.

### 5.8 Incremental embedding & indexing
- **FR-23** Only chunks from changed documents are re-embedded; unchanged chunks
  are reused.
- **FR-24** The embedding model used at index time and at query time **must
  match** (same model, same dimension). This is verified and pinned in config.
- **FR-25** Indexing updates both the dense vector index and the keyword/BM25
  index together, and is idempotent.

### 5.9 Retrieval
- **FR-26** Retrieval is hybrid (dense + BM25) with **metadata filtering**
  (e.g. restrict to an authority, regulation, or version).
- **FR-27** A cross-encoder reranks the final candidate set before it reaches the
  answer layer.

### 5.10 Deferred
- **FR-28** Vector compression/quantization (the "TurboVec" approach) is
  implemented only when the collection approaches millions of chunks and memory
  is a real bottleneck. It is out of scope until then.

---

## 6. Non-functional requirements

- **NFR-1 Reliability:** every external call (download, DB, embed) has timeout,
  retry-with-backoff, and a defined fallback. No silent failures.
- **NFR-2 Idempotency:** re-running any stage produces no duplicates and no corruption.
- **NFR-3 Single source of truth:** the registry is authoritative; storage and
  indexes are derived from it and reconcilable to it.
- **NFR-4 Observability / step timing:** each stage records its own duration,
  item counts (fetched / accepted / rejected / changed / embedded), and outcome,
  so a run can be inspected step by step ("where did time go, what got dropped, why").
- **NFR-5 Compute efficiency:** incremental processing means a normal run touches
  only changed documents; a full re-index is an explicit, separate action.
- **NFR-6 Security:** source allow-list + HTTPS verification; secrets via env;
  all ingest actions logged.
- **NFR-7 Testing:** every script/module has unit tests for its main path and
  failure paths; one integration test asserts the **flow runs in the expected
  order** (FR-1 → FR-27) and that gates actually block bad input. Existing tests
  stay green.

---

## 7. Suggested data model (registry)

- `regulations` (regulation_id, authority, title, region)
- `documents` (doc_id, regulation_id, version, amendment_series, content_hash,
  source_url, file_path, fetched_at, status)
- `chunks` (chunk_id, doc_id, chunk_type, section_id, heading_path, page,
  metadata json, text, embedding_ref, content_hash)
- `ingest_log` (run_id, stage, item, outcome, reason, duration_ms, timestamp)

---

## 8. Phasing

- **Phase 1 — Acquisition + validation + single store + tracker** (FR-1…FR-13).
  This alone fixes "scattered, mock-only, unvalidated".
- **Phase 2 — Registry + parsing + structure-aware chunking + metadata**
  (FR-14…FR-22).
- **Phase 3 — Incremental embed/index + hybrid retrieval + rerank**
  (FR-23…FR-27).
- **Deferred — vector compression** (FR-28) when scale demands it.

Each phase ships with its unit tests and updates the integration flow test.

---

## 9. Success metrics

- 0 accepted documents outside the canonical store.
- 0 unvalidated/synthetic files in the knowledge base.
- A scheduled run completes automatically with no manual step.
- Re-running with no upstream change embeds 0 chunks (proves incremental).
- Retrieval recall@5 / MRR on a labeled set improves or holds after structure-aware chunking.
- Every run produces a per-stage timing + counts report.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scheduler silently not running (no worker) | Health check + visible status (NFR-4, FR-4) |
| Synthetic/placeholder files pollute corpus | Validation gate + quarantine (FR-5…FR-8) |
| Raw-byte dedup causes false revisions | Hash normalized text (FR-13) |
| Index/query embedding mismatch | Pin + verify one model/dimension (FR-24) |
| Over-engineering (e.g. premature compression) | Defer FR-28; build only what scale needs |
| Losing legacy series on revision | Supersede, never overwrite (FR-15) |

---

## 11. Open questions

- Quarterly cadence confirmed, or per-authority cadence?
- Full expected-coverage list per market (defines the gap report)?
- Hosting / data-residency constraints (likely EU/on-prem)?
