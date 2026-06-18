# Retrieval + evaluation artifacts (v3.2)

## Active (used by backend / hybrid retrieval)

| File | Role |
|------|------|
| `output/regulation_chunks.json` | Hierarchical chunks (BM25 corpus + metadata) |
| `output/regulation_embeddings.json` | Nomic dense vectors (Git LFS) |
| `output/markdown/` | OCR markdown sources |

## Evaluation reports

| Path | Contents |
|------|----------|
| `output/evaluation/current/` | Latest RAGAS run + PNG dashboards |
| `output/evaluation/archive/v3_1/` | Pre–v3.2 corpus eval (1,572 chunks, proxy answers) |
| `output/evaluation/archive/v3_2_snapshots/` | Point-in-time copies before re-chunk fixes |

## Model stack (see `config.py`)

Dense: `nomic-ai/nomic-embed-text-v1.5` → BM25 + cosine → RRF → `BAAI/bge-reranker-v2-m3`
