# Hugging Face Docker Space — Regulation Registry (session auth + confidential upload)
# Corpus DB is fetched from GitHub LFS at build time (keeps HF Space repo under 1 GB).
# Generate bundle locally: python scripts/hf_export_db.py && git push origin main

FROM python:3.11-slim

WORKDIR /app

ARG CORPUS_GIT_REF=main
ARG CORPUS_GIT_URL=https://github.com/Sharan099/safety-assistant.git

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    HF_HOME=/app/.cache/huggingface \
    DATABASE_URL=sqlite:////app/var/safety_registry.db \
    HF_SPACE=true \
    DISABLE_SCHEDULER=true \
    SKIP_WORKER_HEALTH=true \
    CRAWL_ON_STARTUP=false \
    CRAWL_MOCK=true \
    USE_MOCK_EMBEDDINGS=false \
    EMBED_BACKEND=transformers \
    ENABLE_GATEWAY=true \
    ENABLE_PROMETHEUS=false \
    RUN_SELFTEST_ON_STARTUP=false \
    SESSION_COOKIE_SECURE=true \
    SESSION_COOKIE_SAMESITE=none \
    OMP_NUM_THREADS=4 \
    KMP_DUPLICATE_LIB_OK=TRUE

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git git-lfs \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend_requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1.0" \
    && pip install -r /app/requirements.txt -r /app/backend_requirements.txt \
    && pip install aiosqlite>=0.19.0

COPY app/ /app/app/
COPY api/ /app/api/
COPY database/ /app/database/
COPY parser/ /app/parser/
COPY registry/ /app/registry/
COPY vectorization/ /app/vectorization/
COPY scheduler/ /app/scheduler/
COPY backend/ /app/backend/
COPY regulation_discovery/ /app/regulation_discovery/
COPY monitoring/ /app/monitoring/
COPY alembic.ini /app/alembic.ini
COPY coverage_expected.yaml /app/coverage_expected.yaml
COPY scripts/seed_auth_users.py /app/scripts/seed_auth_users.py
COPY docker/hf_entrypoint.sh /app/docker/hf_entrypoint.sh

# Fetch pre-built SQLite corpus from GitHub LFS (not stored in HF Space git).
RUN mkdir -p /app/var /app/storage/confidential/uploads /app/.cache/huggingface \
    && git lfs install \
    && git clone --depth 1 --branch "${CORPUS_GIT_REF}" --filter=blob:none --sparse "${CORPUS_GIT_URL}" /tmp/corpus-src \
    && cd /tmp/corpus-src \
    && git sparse-checkout set data/hf/safety_registry.db \
    && git lfs pull \
    && test -f data/hf/safety_registry.db \
    && cp data/hf/safety_registry.db /app/var/safety_registry.db \
    && rm -rf /tmp/corpus-src \
    && python -c "\
import sqlite3, pathlib; \
p = pathlib.Path('/app/var/safety_registry.db'); \
assert p.stat().st_size > 1_000_000, 'corpus db missing or too small'; \
c = sqlite3.connect(p); \
n = c.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]; \
c.close(); \
assert n > 100, f'chunks table empty (got {n})'" \
    && chmod +x /app/docker/hf_entrypoint.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/v1/health" || exit 1

ENTRYPOINT ["/app/docker/hf_entrypoint.sh"]
