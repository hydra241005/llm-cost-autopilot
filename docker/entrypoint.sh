#!/usr/bin/env bash
set -euo pipefail

# Simple production entrypoint: wait for dependencies, run migrations, validate env, then exec CMD
STATUS=0
log() { echo "[entrypoint] $*"; }

validate_env() {
  local missing=0
  for v in DATABASE_URL REDIS_URL; do
    if [ -z "${!v-}" ]; then
      echo "Missing required env var: $v" >&2
      missing=1
    fi
  done
  if [ "$missing" -eq 1 ]; then
    return 1
  fi
  return 0
}

wait_for_tcp() {
  local host=$1 port=$2 timeout=${3:-30}
  local start_ts=$(date +%s)
  while true; do
    if bash -c "</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      return 0
    fi
    now=$(date +%s)
    if [ $((now - start_ts)) -ge $timeout ]; then
      return 1
    fi
    sleep 1
  done
}

run_migrations() {
  if command -v alembic >/dev/null 2>&1; then
    log "Running alembic migrations"
    alembic upgrade head || return 1
  else
    if python -c "import alembic" >/dev/null 2>&1; then
      python -m alembic upgrade head || return 1
    else
      log "No alembic found; skipping migrations"
    fi
  fi
  return 0
}

trap 'STATUS=$?; log "Received signal, exiting with $STATUS"; exit $STATUS' TERM INT

log "Validating required env vars"
validate_env || exit 2

# Wait for Postgres (parse from DATABASE_URL if possible)
if [ -n "${DATABASE_URL-}" ]; then
  # try to extract host and port
  if echo "$DATABASE_URL" | grep -q "@"; then
    host_port=$(echo "$DATABASE_URL" | sed -E 's#.+@([^/]+).+#\1#')
    host=$(echo "$host_port" | sed -E 's#:(.*)##')
    port=$(echo "$host_port" | sed -nE 's#.*:([0-9]+).*#\1#p')
    port=${port:-5432}
    log "Waiting for Postgres at $host:$port"
    if ! wait_for_tcp "$host" "$port" 60; then
      echo "Timed out waiting for Postgres at $host:$port" >&2
      exit 3
    fi
  fi
fi

# Wait for Redis
if [ -n "${REDIS_URL-}" ]; then
  # REDIS_URL may be redis://host:port
  host_port=$(echo "$REDIS_URL" | sed -E 's#^[^:]+://([^/]+).*#\1#')
  host=$(echo "$host_port" | sed -E 's#:(.*)##')
  port=$(echo "$host_port" | sed -nE 's#.*:([0-9]+).*#\1#p')
  port=${port:-6379}
  log "Waiting for Redis at $host:$port"
  if ! wait_for_tcp "$host" "$port" 30; then
    echo "Timed out waiting for Redis at $host:$port" >&2
    exit 4
  fi
fi

run_migrations || exit 5

log "Entrypoint checks complete; exec: $*"
exec "$@"
