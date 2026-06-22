---
title: Passive Safety Assistant API
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Passive Safety Assistant — Backend API

FastAPI backend for the [Passive Safety Assistant](https://safety-assistant-tan.vercel.app/) frontend.

- **Frontend:** [safety-assistant-tan.vercel.app](https://safety-assistant-tan.vercel.app/)
- **Health:** `GET /api/v1/health`
- **Chat:** `POST /api/v1/chat`
- **Docs:** `/docs`

Hybrid retrieval (Nomic embeddings + BM25 → RRF → cross-encoder rerank) with Groq LLM answers.

## Environment variables (Settings → Variables and secrets)

Set these in the Space **Settings → Variables and secrets** before the app will work:

| Variable | Required | Example / notes |
|----------|----------|-----------------|
| `GROQ_API_KEY` | **Yes** | `gsk_...` from [Groq console](https://console.groq.com) |
| `CORS_ORIGINS` | **Yes** | `https://safety-assistant-tan.vercel.app` |
| `GROQ_MODEL` | Recommended | `llama-3.3-70b-versatile` |
| `ENABLE_HARD_METADATA_FILTER` | Recommended | `true` |
| `EMBEDDING_MODEL` | Recommended | `nomic-ai/nomic-embed-text-v1.5` |
| `EMBEDDING_TRUST_REMOTE_CODE` | Recommended | `true` (required for Nomic) |
| `RERANKER_MODEL` | Recommended | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_KIND` | **Yes** | `crossencoder` — must match model (see below) |
| `ENABLE_RERANKER` | Recommended | `true` |
| `EMBEDDING_REVISION` | Optional | Git ref only (`main` or commit hash). **Do not set to `true`** — that is `EMBEDDING_TRUST_REMOTE_CODE` |

**Reranker pairing** — `RERANKER_KIND` is the *loader type*, not the model name:

| `RERANKER_MODEL` | `RERANKER_KIND` |
|------------------|-----------------|
| `BAAI/bge-reranker-v2-m3` | `crossencoder` |
| `jinaai/jina-reranker-v3` | `jina` |

Wrong pairing (e.g. Jina model + `crossencoder` kind, or putting the model id in `RERANKER_KIND`) causes `Rerank predict failed` and falls back to RRF scores only.
| `ENABLE_PROMETHEUS_METRICS` | Recommended | `false` |
| `RUN_SELFTEST_ON_STARTUP` | Recommended | `false` (faster cold start on free tier) |
| `APP_DB_PATH` | Optional | `/app/var/app.db` (ephemeral on HF — feedback resets on rebuild) |
| `FEEDBACK_DASHBOARD_KEY` | Optional | Secret for admin `/dashboard` on Vercel |
| `HF_TOKEN` | Optional | Hugging Face token for faster model downloads |

**Vercel frontend** must set:

```
NEXT_PUBLIC_API_URL=https://sharan099-passive-safety-assistant.hf.space/api/v1
```

(Add in Vercel → Project → Settings → Environment Variables, then redeploy.)
