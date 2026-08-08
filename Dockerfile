# Passive Safety RAG API (Qdrant + FastAPI)
FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY ingestion ./ingestion
COPY retrieval ./retrieval
COPY generation ./generation
COPY api ./api
COPY eval ./eval
COPY app ./app
COPY observability ./observability
COPY config ./config

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY ingestion ./ingestion
COPY retrieval ./retrieval
COPY generation ./generation
COPY api ./api
COPY eval ./eval
COPY app ./app
COPY observability ./observability
COPY config ./config

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000
ENV RERANK_PROVIDER=none

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
