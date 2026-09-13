# ADR-0030 — Summary-augmented chunking: document identity in the retrieval representation, never in the evidence

Status: accepted as a measured option · Date: 2026-09-13 · Related: 0029 (schema), README "Measured results"

## Problem
Regulations in the corpus share wording almost verbatim across documents: UN R94 §5.2.3 and UN R95 §5.3.1 (no door shall open), R94 §5.2.7 and R95 §5.3.6 (30 g/min fuel leakage), R16 §2.32 and R129 §2.11 (ISOFIX anchorage definition). A chunk-level retriever returns a semantically right paragraph from the wrong document — measured before this change as DRM@1 8.1 % on `regulatory_v2` and 38.9 % on the dedicated `document_mismatch_v1` benchmark (full pipeline). The chunk header already carries `regulation › breadcrumb`, but it is version-independent and says nothing about *what the document governs* (frontal vs. side impact, adult belts vs. child restraints), which is exactly what these queries turn on.

## Decision
Every chunk keeps two representations:

- `chunks.content` — the original regulatory text. The only text an answer may quote or cite. Unchanged.
- `chunks.retrieval_text` — `SOURCE DOCUMENT` identity block (deterministic columns: regulation, title, kind, authority, jurisdiction, version label, series, revision, dates; unknown values omitted) + `DOCUMENT SUMMARY` (one generated per document version) + `content`. Indexed only.

The summary is generated once per (artifact sha256, prompt version, model) by the configured LLM provider from a bounded excerpt (front matter, section outline, first pages) and stored in `document_summaries` with `status` READY | FAILED | SKIPPED. Failure or a provider not cleared for the document's data class degrades to identity block + content — a version is never blocked and never left without a searchable representation.

Both retrieval legs get a second index version rather than a rewrite: `chunk_embeddings.representation` (`content` | `sac_v1` | `sac_v2`) with one partial HNSW index each, and one BM25 index per representation. `RETRIEVAL_REPRESENTATION` selects which the query side uses; `SAC_ENABLED` makes ingestion build the SAC representation; `safety-assistant reindex` builds it for an existing corpus (resumable, idempotent, per-version commit). Exact-clause lookup, the temporal/version filter, the reranker input (chunk content) and the evidence bundle are untouched.

## Alternatives rejected
- Rewriting `content` with the context: contaminates evidence and citations; forbidden by the grounding invariant.
- Query-time summarisation: LLM cost per query for information that is a property of the document.
- Per-chunk LLM context (Anthropic-style contextual retrieval): ~22,000 calls vs. 18 here, and inconsistent context across chunks of one document.
- Replacing hybrid retrieval with a document-first router: the exact leg and RRF already work; the experiment isolates one variable.

## Outcome (measured 2026-09-13, README "Measured results")
Two representations were built: `sac_v1` (full identity block + summary + chunk) and `sac_v2` (one identity line + first summary sentence + chunk). fastembed truncates MiniLM input at 128 tokens, so the `sac_v1` prefix alone fills the dense window and its vector is a document vector; `sac_v2` keeps the chunk in the window. On identical queries and labels, `sac_v2` improved every document-level metric on all three sets (v2 DRM@1 0.081 → 0.069, v1 0.093 → 0.047, twin-clause benchmark 0.389 → 0.333) at −0.008 passage MRR / −0.016 R@10 on the broad set; `sac_v1` is the strongest on the twin-clause benchmark (DRM@1 0.306, passage MRR +0.091) but regresses the broad set (R@5 −0.039). Leg ablations showed the BM25 leg carries the hard cases and the dense `sac_v1` vector acts as a harmless document prior. Decision: `sac_v2` recommended (`RETRIEVAL_REPRESENTATION=sac_v2`, built by `reindex` / `SAC_ENABLED`), `sac_v1` experimental, baseline untouched and still the code default. Giving the cross-encoder the document line with the chunk (`RETRIEVAL_RERANK_WITH_CONTEXT`) halves the twin-clause mismatch rate again (0.333 → 0.194) but costs 0.036 passage MRR on the human-written set, so it is an option, not a default. A first summary round from the free "auto" route produced reasoning dumps for 10 of 16 documents; the validator rejects those and `SUMMARY_MODEL` selects a plain instruct model.

## Consequences
- SAC embeddings carry version identity, so the cross-version embedding reuse that the baseline enjoys (identical clauses across revisions) does not apply to `sac_v1`; each version is embedded once. Storage doubles for the vector table.
- The identity block repeats document words in every chunk of that document; the existing per-version cap and sha-dedup in `retrieval/service.py` bound domination, and `RetrievalResult.document_distribution()` makes it inspectable.
- Whether SAC becomes the default is decided by the A/B in README "Measured results", not by this record.
