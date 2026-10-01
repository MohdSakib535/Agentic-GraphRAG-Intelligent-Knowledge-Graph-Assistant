"""Celery tasks: document ingestion and evaluation runs."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from celery import Task
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from app.core.config import get_settings
from app.core.errors import AppError, EmbeddingError, Neo4jUnavailable
from app.core.logging import get_logger, tenant_id_ctx
from app.db.neo4j import ensure_schema_sync, get_sync_driver
from app.db.postgres import sync_session_scope
from app.db.redis import invalidate_tenant_cache_sync
from app.graph.builder import GraphBuilder
from app.graph.repository import GraphWriter
from app.ingestion.embedding import build_embedder
from app.ingestion.entity_resolver import EntityResolver
from app.ingestion.loader import FileStorage
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.relationship_extractor import build_graph_extractor
from app.llm.client import build_llm_client
from app.models.document import Document, DocumentStatus
from app.models.job import STAGE_PROGRESS, IngestionJob, JobStage, JobStatus
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

TRANSIENT = (Neo4jUnavailable, ServiceUnavailable, SessionExpired, EmbeddingError, ConnectionError)
_schema_ready = False


def _update_job(job_id: uuid.UUID, **values: Any) -> None:
    with sync_session_scope() as db:
        job = db.get(IngestionJob, job_id)
        if job is None:
            return
        for key, value in values.items():
            setattr(job, key, value)


def build_pipeline() -> IngestionPipeline:
    global _schema_ready
    settings = get_settings()
    driver = get_sync_driver(settings)
    if not _schema_ready:
        ensure_schema_sync(driver, settings)
        _schema_ready = True
    writer = GraphWriter(driver, settings)
    llm = build_llm_client(settings)
    embedder = build_embedder(settings)
    return IngestionPipeline(
        settings=settings,
        storage=FileStorage(settings.upload_dir),
        embedder=embedder,
        extractor_factory=lambda: build_graph_extractor(settings, llm),
        resolver=EntityResolver(settings, embedder=embedder, lookup=writer, llm=llm),
        builder=GraphBuilder(writer),
    )


@celery_app.task(bind=True, name="app.workers.tasks.ingest_document", max_retries=3)
def ingest_document(self: Task, job_id: str) -> dict[str, Any]:
    job_uuid = uuid.UUID(job_id)
    settings = get_settings()
    with sync_session_scope() as db:
        job = db.get(IngestionJob, job_uuid)
        if job is None:
            logger.warning("ingestion_job_missing", extra={"job_id": job_id})
            return {"status": "missing"}
        document = db.get(Document, job.document_id)
        if document is None or str(document.tenant_id) != str(job.tenant_id):
            job.status, job.stage, job.error_code = JobStatus.FAILED, JobStage.FAILED, "DOCUMENT_NOT_FOUND"
            return {"status": "failed"}
        tenant_id, document_id = str(job.tenant_id), str(document.id)
        filename, file_type, storage_path = document.filename, document.file_type, document.storage_path
        job.status, job.attempts = JobStatus.RUNNING, job.attempts + 1
        job.started_at = job.started_at or datetime.now(UTC)
        job.celery_task_id = self.request.id
        document.status = DocumentStatus.PROCESSING
    tenant_id_ctx.set(tenant_id)
    history: list[dict[str, Any]] = []

    def on_stage(stage: str, detail: dict[str, Any]) -> None:
        history.append({"stage": str(stage), "at": datetime.now(UTC).isoformat(), **detail})
        _update_job(job_uuid, stage=str(stage), progress=STAGE_PROGRESS.get(stage, 0), stage_history=list(history))

    try:
        result = build_pipeline().run(tenant_id=tenant_id, document_id=document_id, filename=filename,
                                      file_type=file_type, storage_path=storage_path, on_stage=on_stage)
    except TRANSIENT as exc:
        if self.request.retries < self.max_retries:
            _update_job(job_uuid, error_code="RETRYING", error_message=f"Transient failure: {type(exc).__name__}")
            raise self.retry(exc=exc, countdown=2 ** (self.request.retries + 2)) from exc
        _fail(job_uuid, "SERVICE_UNAVAILABLE", "A required service was unavailable during processing", history)
        return {"status": "failed"}
    except AppError as exc:
        _fail(job_uuid, exc.code, exc.message, history)
        return {"status": "failed"}
    except Exception:
        logger.exception("ingestion_failed", extra={"job_id": job_id})
        _fail(job_uuid, "DOCUMENT_PROCESSING_FAILED", "Document processing failed", history)
        return {"status": "failed"}

    with sync_session_scope() as db:
        job = db.get(IngestionJob, job_uuid)
        document = db.get(Document, uuid.UUID(document_id))
        if job is not None:
            job.status, job.stage, job.progress = JobStatus.COMPLETED, JobStage.COMPLETED, 100
            job.finished_at = datetime.now(UTC)
            job.stats = result.stats()
            job.error_code = job.error_message = None
        if document is not None:
            document.status = DocumentStatus.COMPLETED
            document.title = (result.title or document.title or "")[:500]
            document.page_count = result.page_count
            document.chunk_count = result.chunks
            document.entity_count = result.entities
            document.relationship_count = result.relationships
            document.metadata_ = result.metadata
            document.error_message = None
    invalidate_tenant_cache_sync(settings.redis_url, tenant_id)
    return {"status": "completed", **result.stats()}


def _fail(job_id: uuid.UUID, code: str, message: str, history: list[dict[str, Any]]) -> None:
    with sync_session_scope() as db:
        job = db.get(IngestionJob, job_id)
        if job is None:
            return
        job.status, job.stage = JobStatus.FAILED, JobStage.FAILED
        job.error_code, job.error_message = code, message[:1000]
        job.finished_at = datetime.now(UTC)
        job.stage_history = [*history, {"stage": "FAILED", "at": datetime.now(UTC).isoformat(), "error": code}]
        document = db.get(Document, job.document_id)
        if document is not None:
            document.status = DocumentStatus.FAILED
            document.error_message = message[:1000]


@celery_app.task(name="app.workers.tasks.run_evaluation")
def run_evaluation(run_id: str) -> dict[str, Any]:
    from app.services.evaluation_service import execute_evaluation_run

    return asyncio.run(execute_evaluation_run(uuid.UUID(run_id)))
