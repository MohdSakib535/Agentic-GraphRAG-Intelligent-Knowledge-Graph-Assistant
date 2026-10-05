#!/bin/sh
set -e
role="${1:-api}"
case "$role" in
  api)
    echo "Running database migrations..."
    alembic upgrade head
    # Prometheus multiprocess mode: one shared sample directory for all uvicorn workers.
    export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus}"
    rm -rf "$PROMETHEUS_MULTIPROC_DIR" && mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers "${API_WORKERS:-2}" --proxy-headers --forwarded-allow-ips="*"
    ;;
  worker)
    exec celery -A app.workers.celery_app worker --loglevel="${LOG_LEVEL:-INFO}" \
      -Q ingestion,evaluation,default --concurrency="${WORKER_CONCURRENCY:-2}"
    ;;
  beat)
    exec celery -A app.workers.celery_app beat --loglevel="${LOG_LEVEL:-INFO}" --schedule /tmp/celerybeat-schedule
    ;;
  migrate)
    exec alembic upgrade head
    ;;
  seed)
    exec python -m app.scripts.seed_demo
    ;;
  *)
    exec "$@"
    ;;
esac
