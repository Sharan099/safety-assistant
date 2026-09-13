# Developer entry points. Everything runs through uv.
.PHONY: setup db migrate ingest api worker users test lint types eval reindex eval-sac eval-judged frontend e2e load docker up down

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
worker:           ## run the ingestion queue worker (uploads + queued registry sources)
	uv run safety-assistant worker
users:            ## seed a development engineer (DEV_LOGIN_ENABLED=true): make users EMAIL=you@example.com
	uv run safety-assistant users add --email $(EMAIL) --name "$(EMAIL)" --role engineer --workspace default
test:             ## rebuilt test suite (unit, golden, security, integration, e2e, evaluation, regression)
	uv run pytest -q
lint:
	uv run ruff check src tests scripts migrations && uv run ruff format --check src tests scripts
types:
	uv run mypy src scripts/eval scripts/maintenance
eval:             ## per-leg retrieval evaluation -> evals/results/
	uv run safety-assistant eval-retrieval --dataset evals/datasets/regulatory_v2.yaml
reindex:          ## build the summary-augmented (sac_v1) index for the existing corpus; resumable
	uv run safety-assistant reindex
eval-sac:         ## baseline vs SAC A/B on the document-mismatch benchmark and regulatory_v2
	uv run python scripts/eval/sac_ab.py --datasets evals/datasets/document_mismatch_v1.yaml evals/datasets/regulatory_v2.yaml
eval-judged:      ## end-to-end answers + deterministic metrics + RAGAS/DeepEval judges (needs `uv sync --extra eval` and an LLM)
	uv run python scripts/eval/judged.py --dataset evals/datasets/regulatory_v2.yaml --ragas --deepeval
frontend:         ## type-check, lint and build the Next.js app
	cd frontend && npm ci && npm run typecheck && npm run lint && npm run build
e2e:              ## Playwright flows (API on :8010 with DEV_LOGIN_ENABLED=true and a running worker)
	cd frontend && npx playwright test
load:             ## closed-loop load test against a running API
	uv run python scripts/eval/load_test.py --users 5 --seconds 60
docker:           ## build the hardened image
	docker build -f infra/docker/Dockerfile -t safety-assistant:local .
up:               ## full local stack (api + postgres + minio)
	docker compose -f infra/docker/compose.yaml up --build
down:
	docker compose -f infra/docker/compose.yaml down
