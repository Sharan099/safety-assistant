# Data lineage

## Stores

| Store | Content | Mutability |
|---|---|---|
| `knowledge/**/*.pdf` (gitignored) + registry hashes | licensed source PDFs | immutable; hash-checked before every read |
| Blob store (`file://data/artifacts` dev, `s3://` prod) | source bytes and figure images, content-addressed `sha256/<ab>/<sha>` | write-once |
| PostgreSQL | regulations, versions, sections, tables, figures, cross-refs, chunks, embeddings, runs, events, traces, feedback, eval cases | versions are append-only; sections/chunks are rewritten per version when structure changes |

## Lifecycle and keys

```
DISCOVERED  registry entry → regulations upsert, regulation_versions row (status)
DOWNLOADED  local copy (or SSRF-safe fetch with ETag/If-Modified-Since) → source_artifacts.storage_uri
VALIDATED   magic bytes, size, page count, SHA-256 == registry
PARSED      DocumentParser → pages/tables/figures + extraction_report (per-page fault isolation)
            key: source_sha256 + parser_version + parser_config_hash
NORMALIZED  clause tree, definitions, cross-refs, cover-page amendments; parsed_hash
CHUNKED     structural chunks; key: parsed_hash + chunker_version + chunker_config_hash
INDEXED     embeddings; key: chunk_sha256 + model_version + dimensions (reuse across versions)
VERIFIED    chunks == embeddings, citation labels present
ACTIVE      previous ACTIVE version → SUPERSEDED (valid_to = new valid_from) in the same transaction
```

Re-running an unchanged source: `SKIPPED_UNCHANGED` (1–2 s). Structure unchanged but config unchanged: sections/chunks kept. Structure changed: rewritten; embeddings for identical chunk text are reused from a pre-rewrite snapshot and from sibling versions.

## Versions recorded on every regulation_version

`parser_name/version/config_hash`, `parsed_hash`, `chunker_version/config_hash`, `index_schema_version`, `amendments` (cover page), `extraction_report` (QA), `activated_at`, `superseded_by_id`.

## Every answer records (query_traces)

principal (id, never a secret), scopes, query, parsed scope, plan (intent, route, rewrites, attempts, budgets), every fused candidate with leg ranks and scores, selected evidence, answer/claims/abstain reason, validation report, versions (embedding model, reranker, prompt version, LLM model), per-stage latency, tokens.

## Freshness

`ingestion_runs.stats.freshness_lag_days_from_publication` and the gauge `sa_freshness_lag_days{regulation}`; `GET /api/v1/regulations/{key}/versions?as_of=` resolves the version in force on a date.
