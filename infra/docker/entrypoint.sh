#!/bin/sh
# Container entrypoint. `api` serves; `migrate` applies schema; `ingest` runs the
# registry ingestion; anything else is executed verbatim (e.g. a shell for debugging).
set -eu
case "${1:-api}" in
  api)
    exec uvicorn safety_assistant.api.main:app --host 0.0.0.0 --port 8010 \
      --workers "${WEB_CONCURRENCY:-2}" --timeout-keep-alive 15 --no-server-header ;;
  migrate)
    exec alembic -c migrations/alembic.ini upgrade head ;;
  ingest)
    shift; exec python -m safety_assistant.cli ingest "$@" ;;
  *)
    exec "$@" ;;
esac
