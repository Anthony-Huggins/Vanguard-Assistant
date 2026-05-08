#!/bin/sh
# api container entrypoint — runs migrations then starts uvicorn.
#
# Alembic is idempotent so it's safe to run on every boot (it short-circuits
# when the DB is already at head).  We don't auto-ingest: documents land in
# the chroma container via ``docker compose run --rm ingest`` (one-time).

set -e

echo "[api] applying alembic migrations..."
python -m alembic upgrade head

echo "[api] starting uvicorn..."
exec uvicorn apps.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --proxy-headers
