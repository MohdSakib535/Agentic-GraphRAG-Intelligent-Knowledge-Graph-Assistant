"""Celery application (Redis broker + result backend)."""

from __future__ import annotations

from celery import Celery
from celery.signals import after_setup_logger, worker_process_init, worker_process_shutdown

from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()

celery_app = Celery("agentic_graphrag", broker=settings.broker_url, backend=settings.result_backend,
                    include=["app.workers.tasks"])
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,  # re-deliver if a worker dies mid-task (tasks are idempotent)
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # long-running tasks: fair dispatch
    task_time_limit=60 * 30,
    task_soft_time_limit=60 * 25,
    result_expires=60 * 60 * 24,
    broker_connection_retry_on_startup=True,
    task_default_queue="default",
    task_routes={"app.workers.tasks.ingest_document": {"queue": "ingestion"},
                 "app.workers.tasks.sync_connector": {"queue": "ingestion"},
                 "app.workers.tasks.run_evaluation": {"queue": "evaluation"}},
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=False,
)


if settings.connector_sync_interval_minutes > 0:
    celery_app.conf.beat_schedule = {
        "sync-connectors": {"task": "app.workers.tasks.sync_all_connectors",
                            "schedule": settings.connector_sync_interval_minutes * 60.0},
    }


@after_setup_logger.connect
def _setup_logging(**_: object) -> None:
    configure_logging(settings.log_level, settings.log_json)


@worker_process_init.connect
def _setup_tracing(**_: object) -> None:
    from app.core.telemetry import setup_tracing

    setup_tracing(settings, component="worker")


@worker_process_shutdown.connect
def _shutdown(**_: object) -> None:
    from app.db.neo4j import close_sync_driver

    close_sync_driver()
