#!/bin/sh
set -e

export DATABASE_URL="${DATABASE_URL:-sqlite:////app/var/safety_registry.db}"
export PYTHONPATH=/app
export PORT="${PORT:-7860}"

mkdir -p /app/var /app/storage/confidential/uploads

# Idempotent auth seed (skips existing users unless AUTH_SEED_PASSWORD changes with --reset)
python /app/scripts/seed_auth_users.py || true

echo "Starting Regulation Registry on port ${PORT} (db=${DATABASE_URL})"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
