# Hugging Face Docker Space — AutoSafety RAG backend
# Space: https://huggingface.co/spaces/sharan099/Passive_safety_assistant
# Frontend: https://safety-assistant-tan.vercel.app/
# Must listen on port 7860 (HF Spaces requirement).

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    APP_DB_PATH=/app/var/app.db

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1.0" \
    && pip install -r /app/requirements.txt

COPY config.py /app/config.py
COPY backend /app/backend
COPY ingestion /app/ingestion
COPY output/regulation_chunks.json output/ingest_manifest.json /app/output/

# Embeddings (~466 MB). Prefer COPY from Space repo (Git LFS).
# Fallback: pull from GitHub if the pointer file slipped through.
COPY output/regulation_embeddings.json /app/output/regulation_embeddings.json
RUN python -c "\
import json, pathlib; \
p=pathlib.Path('/app/output/regulation_embeddings.json'); \
d=json.loads(p.read_text()); \
n=len(d.get('embeddings',{})); \
assert n>1000, f'embeddings invalid (got {n}) — use Git LFS for regulation_embeddings.json'"

RUN mkdir -p /app/var /app/.cache/huggingface

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f "http://127.0.0.1:${PORT}/api/v1/health" || exit 1

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
