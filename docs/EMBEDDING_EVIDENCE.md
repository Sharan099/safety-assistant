# Live `safety_registry.db` embedding model evidence

**Conclusion: vectors were produced with `nomic-ai/nomic-embed-text-v1.5` (768 dimensions), not BGE-large (1024).**

## Measurements (2026-06-26)

| Check | Result |
|-------|--------|
| Total chunks | 7,650 |
| Chunks with embeddings | 7,650 (100%) |
| Parsed vector dimension | **768** (sample of 100: all 768) |
| Vector L2 norm | ~1.0 (unit-normalized, consistent with SentenceTransformer) |

## Ruled out

| Model | Dimension | Why excluded |
|-------|-----------|--------------|
| `BAAI/bge-large-en-v1.5` | 1024 | DB vectors are 768-d |
| Windows mock embedder (pre-fix) | 1024 | DB vectors are 768-d |

## Supporting configuration

- `.env` sets `EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5`
- Nomic Embed Text v1.5 publishes **768-dimensional** embeddings

## Code alignment (post-pin)

- `registry/embedding_config.py` — single source of truth
- `database/models.py` — `SafeVector(768)`
- `vectorization/embedder.py` — Nomic query/doc prefixes, mock dim 768

**No re-embedding performed** — pin matches existing live index.
