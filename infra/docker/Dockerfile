# syntax=docker/dockerfile:1.7
# Multi-stage, non-root, pinned via uv.lock, no dev tools in the runtime image.
# Build:  docker build -f infra/docker/Dockerfile -t safety-assistant:local .
# Run:    docker run --rm -p 8010:8010 --env-file .env safety-assistant:local

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (layer cache), project code second.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra s3
COPY src ./src
COPY migrations ./migrations
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra s3

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
LABEL org.opencontainers.image.title="safety-assistant" \
      org.opencontainers.image.description="Regulatory knowledge API: versioned ingestion, hybrid retrieval, grounded answers" \
      org.opencontainers.image.source="https://github.com/OWNER/safety-assistant"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH" \
    APP_ENV=production HF_HOME=/models FASTEMBED_CACHE_PATH=/models
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home /app app \
    && mkdir -p /models /data && chown -R app:app /models /data
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/migrations /app/migrations
COPY --chown=app:app knowledge/00_registry /app/knowledge/00_registry
COPY --chown=app:app infra/docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
USER app
# Embedding model is fetched at build time so the runtime never downloads (air-gap friendly, fast readiness).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8010/health/live', timeout=3).status==200 else 1)"
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
