# Contributing

1. `uv sync --extra s3 && docker compose up -d postgres && cp .env.example .env && uv run safety-assistant migrate`
2. Work in a branch; keep the change small and covered: `make lint types test`.
3. Retrieval or chunking changes must include a before/after evaluation (`make eval`) and, if behaviour changes, an ADR under `docs/ADR/` (next number).
4. Never weaken tests to make a refactor pass; never add a fake provider path outside `APP_ENV=test`.
5. Corpus changes go through `knowledge/00_registry/sources.yaml` (hash, dates, license) and `scripts/maintenance/verify_registry.py`.
6. Update `CHANGELOG.md` under *Unreleased*.
