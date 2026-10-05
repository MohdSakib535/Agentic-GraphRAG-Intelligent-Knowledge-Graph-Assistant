"""Incremental Google Drive -> knowledge-base sync (runs in a Celery worker).

New files are ingested, changed files (``modifiedTime`` differs) re-ingested in place, files removed
from the folder are deleted from the knowledge base (graph, chunks, embeddings, stored file).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select

from app.connectors.google_drive import GoogleDriveClient, parse_service_account
from app.core.config import Settings, get_settings
from app.core.crypto import decrypt
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.neo4j import get_sync_driver
from app.db.postgres import sync_session_scope
from app.db.redis import invalidate_tenant_cache_sync
from app.graph.repository import GraphWriter
from app.ingestion.loader import FileStorage, validate_upload
from app.models.connector import Connector
from app.models.document import Document, DocumentStatus
from app.models.job import IngestionJob, JobStage, JobStatus

logger = get_logger(__name__)


def _enqueue(job_ids: list[str]) -> None:
    from app.workers.tasks import ingest_document

    for job_id in job_ids:
        ingest_document.apply_async(args=[job_id], task_id=job_id)


def sync_google_drive(connector_id: uuid.UUID, settings: Settings | None = None,
                      transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    with sync_session_scope() as db:
        connector = db.get(Connector, connector_id)
        if connector is None:
            return {"status": "missing"}
        connector.status, connector.last_error = "RUNNING", None
        tenant_id, folder_id = connector.tenant_id, connector.config["folder_id"]
        access_groups, created_by = list(connector.access_groups or []), connector.created_by
        credentials = decrypt(settings, connector.encrypted_credentials)
    stats = {"created": 0, "updated": 0, "unchanged": 0, "removed": 0, "skipped": [], "failed": []}
    jobs: list[str] = []
    storage = FileStorage(settings.upload_dir)
    client = GoogleDriveClient(settings, parse_service_account(credentials), transport)
    try:
        files, skipped = client.list_files(folder_id, settings.connector_max_files_per_sync)
        stats["skipped"] = skipped[:100]
        with sync_session_scope() as db:
            existing = {d.external_id: d for d in db.execute(
                select(Document).where(Document.connector_id == connector_id, Document.tenant_id == tenant_id)).scalars()}
        seen: set[str] = set()
        for item in files:
            seen.add(item.id)
            doc = existing.get(item.id)
            if doc is not None and doc.external_modified_at == item.modified_time:
                stats["unchanged"] += 1
                continue
            try:
                data = client.download(item)
                validated = validate_upload(item.filename, None, data, settings.max_upload_bytes)
            except AppError as exc:
                stats["failed"].append({"file": item.path, "error": exc.code})
                continue
            with sync_session_scope() as db:
                duplicate = db.execute(select(Document.id).where(
                    Document.tenant_id == tenant_id, Document.checksum == validated.checksum,
                    Document.id != (doc.id if doc else None))).first()
                if duplicate:
                    stats["skipped"].append(f"{item.path} (duplicate of an existing document)")
                    continue
                if doc is None:
                    doc_id = uuid.uuid4()
                    path = storage.save(str(tenant_id), str(doc_id), validated.file_type, validated.data)
                    db.add(Document(id=doc_id, tenant_id=tenant_id, uploaded_by=created_by, filename=validated.filename,
                                    file_type=validated.file_type, content_type=item.mime_type, size_bytes=validated.size,
                                    checksum=validated.checksum, storage_path=path, status=DocumentStatus.PENDING,
                                    metadata_={"drive_path": item.path}, access_groups=access_groups,
                                    source="google_drive", connector_id=connector_id, external_id=item.id,
                                    external_modified_at=item.modified_time))
                    db.flush()
                    stats["created"] += 1
                else:
                    doc_id = doc.id
                    current = db.get(Document, doc_id)
                    assert current is not None
                    if current.file_type != validated.file_type:
                        storage.delete(current.storage_path)
                    current.storage_path = storage.save(str(tenant_id), str(doc_id), validated.file_type, validated.data)
                    current.file_type, current.filename, current.size_bytes = (
                        validated.file_type, validated.filename, validated.size)
                    current.checksum, current.external_modified_at = validated.checksum, item.modified_time
                    current.status, current.access_groups = DocumentStatus.PENDING, access_groups
                    stats["updated"] += 1
                job = IngestionJob(tenant_id=tenant_id, document_id=doc_id, status=JobStatus.QUEUED,
                                   stage=JobStage.QUEUED, stage_history=[], stats={})
                db.add(job)
                db.flush()
                jobs.append(str(job.id))
        # Files that disappeared from the folder are removed from the knowledge base.
        writer = GraphWriter(get_sync_driver(settings), settings)
        for external_id, doc in existing.items():
            if external_id in seen:
                continue
            writer.delete_document(str(tenant_id), str(doc.id))
            storage.delete(doc.storage_path)
            with sync_session_scope() as db:
                stale = db.get(Document, doc.id)
                if stale is not None:
                    db.delete(stale)
            stats["removed"] += 1
        status, error = "IDLE", None
    except AppError as exc:
        status, error = "FAILED", f"{exc.code}: {exc.message}"
        logger.warning("connector_sync_failed", extra={"connector_id": str(connector_id), "code": exc.code})
    finally:
        client.close()
    with sync_session_scope() as db:
        connector = db.get(Connector, connector_id)
        if connector is not None:
            connector.status, connector.last_error = status, error
            connector.last_sync_at = datetime.now(UTC)
            connector.last_sync_stats = stats
    _enqueue(jobs)
    if stats["removed"]:
        invalidate_tenant_cache_sync(settings.redis_url, str(tenant_id))
    return {"status": status, **{k: v for k, v in stats.items() if isinstance(v, int)}}
