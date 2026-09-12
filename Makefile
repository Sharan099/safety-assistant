# Developer entry points. Everything runs through uv.
.PHONY: setup db migrate ingest api test lint types eval load docker up down

setup:            ## install dependencies (incl. s3 extra)
	uv sync --extra s3
db:               ## start PostgreSQL/pgvector (host port 5433)
	docker compose up -d postgres
migrate:          ## apply the safety_assistant schema
	uv run safety-assistant migrate
ingest:           ## ingest every registered source (idempotent)
	uv run safety-assistant ingest
api:              ## run the API on :8010
	uv run uvicorn safety_assistant.api.main:app --port 8010 --reload
test:             ## rebuilt test suite (unit, golden, security, integration, e2e, evaluation, regression)
	uv run pytest -q
lint:
	uv run ruff check src tests scripts migrations && uv run ruff format --check src tests scripts
types:
	uv run mypy src scripts/eval scripts/maintenance
eval:             ## per-leg retrieval evaluation -> evals/results/
	uv run safety-assistant eval-retrieval
load:             ## closed-loop load test against a running API
	uv run python scripts/eval/load_test.py --users 5 --seconds 60
docker:           ## build the hardened image
	docker build -f infra/docker/Dockerfile -t safety-assistant:local .
up:               ## full local stack (api + postgres + minio)
	docker compose -f infra/docker/compose.yaml up --build
down:
	docker compose -f infra/docker/compose.yaml down
