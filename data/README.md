# data/

Everything under here is generated, not authored, and is gitignored
(`data/raw/`, `data/artifacts/`, `data/parquet/`, `data/synthetic/*`).

Regenerate the synthetic benchmark (SCN-001..SCN-010) at any time:

```powershell
docker compose up -d postgres
uv run alembic upgrade head
uv run python scripts/generate_synthetic_dataset.py
```

This writes `data/parquet/<RUN_ID>/*.parquet` (raw signal samples — see
`TRD.md` §8) and `data/synthetic/scenarios.yaml` (ground truth per scenario —
ownership: `packages/analysis/synthetic.py`), and registers the runs in
Postgres. Re-running is idempotent: it clears and reloads the synthetic
project's rows rather than accumulating duplicates.
