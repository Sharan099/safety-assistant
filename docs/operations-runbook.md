# Operations runbook

## Health
- `GET /health/live` — process alive. `GET /health/ready` — 200 only when the database answers, migrations are at head, the embedding model is loaded and at least one version is ACTIVE. `GET /health/deps` — adds LLM status (informational). `GET /metrics` — Prometheus.
- CLI: `uv run safety-assistant ready`.

## Routine tasks
| Task | Command |
|---|---|
| apply migrations | `uv run safety-assistant migrate` (container: `entrypoint.sh migrate`) |
| verify corpus copies | `uv run python scripts/maintenance/verify_registry.py` |
| ingest / re-ingest | `uv run safety-assistant ingest [source_key] [--force]` — idempotent; unchanged sources are no-ops |
| add a new regulation version | add a registry entry (new `version.label`, dates, hash), place the PDF, run ingest; activation supersedes the previous version atomically |
| roll back a bad activation | `UPDATE regulation_versions SET status='SUPERSEDED' … ; UPDATE … SET status='ACTIVE', valid_to=NULL, superseded_by_id=NULL` on the previous version inside one transaction, then `SELECT` to confirm exactly one ACTIVE per regulation; the state machine permits `SUPERSEDED → ACTIVE` |
| evaluate retrieval | `uv run safety-assistant eval-retrieval` |
| load test | `scripts/eval/load_test.py` against a server started with `RATE_LIMIT_PER_MINUTE=0` |

## Deploy / rollback
- Image tags are immutable (`ghcr.io/…:<semver>`); ECS deployment circuit breaker rolls back automatically on failed health checks; manual rollback = re-deploy the previous task definition revision.
- Migrations run as a separate one-shot task before the service update (`entrypoint.sh migrate`). Migrations are additive; downgrade scripts exist (`alembic downgrade -1`) but a schema rollback should be accompanied by an application rollback.

## Backup / restore
- RDS: automated daily snapshots, 14-day PITR (`backup_retention_period = 14`). Restore = point-in-time restore to a new instance, update the `DATABASE_URL` secret, redeploy.
- Artifacts: versioned S3 bucket; objects are content-addressed and never overwritten.
- Corpus PDFs: kept outside git; registry hashes let you verify any restored copy.
- Full rebuild from sources is deterministic: `migrate` + `ingest` regenerates every derived table.

## Capacity notes
- Memory: fastembed model (~90 MB) + BM25 index (~150 MB for 22k chunks) per worker; the API task is sized at 3 GB for 2 workers.
- First request per worker builds the BM25 index (3–5 s at 22k chunks); warm it with a request after deploy or rely on the ALB health check grace period (60 s).
- Ingestion of the 4,301-page manual peaks around 1.5 GB; run ingestion as a one-off task, not inside the API process.

## Dashboards / alerts
`infra/monitoring/grafana-dashboard.json`, `infra/monitoring/prometheus-alerts.yaml` (5xx rate, answer p95, retrieval no-hit spike, citation validation failures, ingestion failures, LLM unavailable). CloudWatch alarms for ALB 5xx and p95 in Terraform.
