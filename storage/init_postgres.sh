#!/usr/bin/env bash
set -euo pipefail

POSTGRES_USER="${POSTGRES_USER:-tatahunts_user}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-tatahunts_password}"
POSTGRES_DB="${POSTGRES_DB:-tatahunts_rag}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-tatahunts_postgres}"

export PGPASSWORD="${POSTGRES_PASSWORD}"

if command -v psql >/dev/null 2>&1; then
  if psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -f storage/postgres_schema.sql; then
    echo "PostgreSQL schema initialized successfully via local psql."
    exit 0
  fi
  echo "Local psql is available but could not connect to ${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}." >&2
  echo "Make sure PostgreSQL server is running and credentials are correct." >&2
fi

if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "${POSTGRES_CONTAINER}"; then
  if docker exec -i "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -f - < storage/postgres_schema.sql; then
    echo "PostgreSQL schema initialized successfully via docker container '${POSTGRES_CONTAINER}'."
    exit 0
  fi
fi

if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
import os
import sys
from pathlib import Path

try:
    import psycopg2
except ModuleNotFoundError:
    print("psql is not available and psycopg2 is not installed.", file=sys.stderr)
    sys.exit(2)

host = os.getenv("POSTGRES_HOST", "localhost")
port = int(os.getenv("POSTGRES_PORT", "5432"))
user = os.getenv("POSTGRES_USER", "tatahunts_user")
password = os.getenv("POSTGRES_PASSWORD", "tatahunts_password")
database = os.getenv("POSTGRES_DB", "tatahunts_rag")

schema_path = Path("storage/postgres_schema.sql")
sql = schema_path.read_text(encoding="utf-8")

conn = psycopg2.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    dbname=database,
)
conn.autocommit = False
try:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
finally:
    conn.close()
PY
  echo "PostgreSQL schema initialized successfully via python/psycopg2."
  exit 0
fi

echo "Failed to initialize schema." >&2
echo "Detected configuration:" >&2
echo "  host=${POSTGRES_HOST} port=${POSTGRES_PORT} db=${POSTGRES_DB} user=${POSTGRES_USER}" >&2
echo "Install one of these options:" >&2
echo "  1) Install PostgreSQL server locally: sudo apt-get install -y postgresql && sudo systemctl enable --now postgresql" >&2
echo "  2) Or run PostgreSQL in container '${POSTGRES_CONTAINER}' (requires docker)" >&2
echo "  3) Ensure DB/user exist and rerun this script" >&2
exit 1
