"""Document lifecycle: upload -> metadata -> ingestion job -> Celery; listing; deletion."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from sqlalchemy import func, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.access import UNRESTRICTED, AccessScope, normalize_groups
from app.core.config import Settings
from app.core.errors import ConflictError, NotFoundError, RedisUnavailable
from app.core.logging import get_logger
from app.db.neo4j import get_sync_driver
from app.db.redis import TenantCache
from app.graph.repository import GraphWriter
from app.ingestion.loader import FileStorage, ValidatedFile
from app.models.document import Document, DocumentStatus
from app.models.job import IngestionJob, JobStage, JobStatus
from app.services.audit import record_audit

logger = get_logger(__name__)


class DocumentService:
    def __init__(self, db: AsyncSession, settings: Settings, cache: TenantCache | None = None) -> None:
        self.db = db
        self.settings = settings
        self.storage = FileStorage(settings.upload_dir)
        self.cache = cache

    async def upload(self, tenant_id: uuid.UUID, user_id: uuid.UUID, file: ValidatedFile,
                     access_groups: list[str] | None = None) -> tuple[Document, IngestionJob]:
        existing = (
            await self.db.execute(
                select(Document).where(Document.tenant_id == tenant_id, Document.checksum == file.checksum)
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise ConflictError(
                f"This file was already uploaded as '{existing.filename}' (id {existing.id})", code="DOCUMENT_ALREADY_EXISTS"
            )
        document_id = uuid.uuid4()
        path = await asyncio.to_thread(self.storage.save, str(tenant_id), str(document_id), file.file_type, file.data)
        document = Document(
            id=document_id, tenant_id=tenant_id, uploaded_by=user_id, filename=file.filename, file_type=file.file_type,
            content_type=file.content_type, size_bytes=file.size, checksum=file.checksum, storage_path=path,
            status=DocumentStatus.PENDING, metadata_={}, access_groups=normalize_groups(access_groups),
        )
        job = IngestionJob(tenant_id=tenant_id, document_id=document_id, status=JobStatus.QUEUED, stage=JobStage.QUEUED,
                           stage_history=[], stats={})
        self.db.add_all([document, job])
        record_audit(self.db, "document.upload", tenant_id=tenant_id, user_id=user_id, resource_type="document",
                     resource_id=str(document_id), details={"filename": file.filename, "size": file.size})
        await self.db.commit()
        await self._enqueue(job)
        return document, job

    async def _enqueue(self, job: IngestionJob) -> None:
        from app.workers.tasks import ingest_document

        try:
            result = await asyncio.to_thread(ingest_document.apply_async, args=[str(job.id)], task_id=str(job.id))
        except Exception as exc:
            logger.error("enqueue_failed", extra={"job_id": str(job.id), "error": type(exc).__name__})
            job.status, job.stage, job.error_code = JobStatus.FAILED, JobStage.FAILED, "QUEUE_UNAVAILABLE"
            job.error_message = "The ingestion queue is unavailable; please retry later"
            await self.db.commit()
            raise RedisUnavailable("The ingestion queue is unavailable") from exc
        job.celery_task_id = result.id
        await self.db.commit()

    async def reprocess(self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID,
                        scope: AccessScope = UNRESTRICTED) -> IngestionJob:
        document = await self.get(tenant_id, document_id, scope)
        document.status = DocumentStatus.PENDING
        job = IngestionJob(tenant_id=tenant_id, document_id=document.id, status=JobStatus.QUEUED,
                           stage=JobStage.QUEUED, stage_history=[], stats={})
        self.db.add(job)
        record_audit(self.db, "document.reprocess", tenant_id=tenant_id, user_id=user_id, resource_type="document",
                     resource_id=str(document_id))
        await self.db.commit()
        await self._enqueue(job)
        return job

    @staticmethod
    def _visible(scope: AccessScope) -> Any:
        denied = [uuid.UUID(d) for d in scope.denied_document_ids]
        return Document.id.not_in(denied) if denied else true()

    async def list(self, tenant_id: uuid.UUID, limit: int, offset: int, status: str | None,
                   scope: AccessScope = UNRESTRICTED) -> tuple[list[Document], int]:
        query = select(Document).where(Document.tenant_id == tenant_id, self._visible(scope))
        count_q = select(func.count()).select_from(Document).where(Document.tenant_id == tenant_id,
                                                                    self._visible(scope))
        if status:
            query = query.where(Document.status == status.upper())
            count_q = count_q.where(Document.status == status.upper())
        rows = (await self.db.execute(query.order_by(Document.created_at.desc()).limit(limit).offset(offset))).scalars()
        total = (await self.db.execute(count_q)).scalar_one()
        return list(rows), int(total)

    async def get(self, tenant_id: uuid.UUID, document_id: uuid.UUID, scope: AccessScope = UNRESTRICTED) -> Document:
        # Tenant + ACL filters in the query itself: another tenant's (or a restricted) id yields 404 - no existence oracle.
        document = (
            await self.db.execute(select(Document).where(Document.id == document_id, Document.tenant_id == tenant_id,
                                                         self._visible(scope)))
        ).scalar_one_or_none()
        if document is None:
            raise NotFoundError("Document not found", code="DOCUMENT_NOT_FOUND")
        return document

    async def latest_job(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> IngestionJob | None:
        return (
            await self.db.execute(
                select(IngestionJob)
                .where(IngestionJob.document_id == document_id, IngestionJob.tenant_id == tenant_id)
                .order_by(IngestionJob.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    async def set_access(self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID,
                         access_groups: list[str]) -> Document:
        document = await self.get(tenant_id, document_id)
        document.access_groups = normalize_groups(access_groups)
        record_audit(self.db, "document.access_changed", tenant_id=tenant_id, user_id=user_id, resource_type="document",
                     resource_id=str(document_id), details={"access_groups": document.access_groups})
        await self.db.commit()
        if self.cache is not None:
            await self.cache.invalidate_tenant(str(tenant_id))  # cached answers were computed under the old ACL
        return document

    async def delete(self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID,
                     scope: AccessScope = UNRESTRICTED) -> None:
        document = await self.get(tenant_id, document_id, scope)
        # Graph first: if Neo4j is down we fail before touching relational state.
        writer = GraphWriter(get_sync_driver(self.settings), self.settings)
        await asyncio.to_thread(writer.delete_document, str(tenant_id), str(document_id))
        await asyncio.to_thread(self.storage.delete, document.storage_path)
        await self.db.delete(document)
        record_audit(self.db, "document.delete", tenant_id=tenant_id, user_id=user_id, resource_type="document",
                     resource_id=str(document_id), details={"filename": document.filename})
        await self.db.commit()
        if self.cache is not None:
            await self.cache.invalidate_tenant(str(tenant_id))

    async def stats(self, tenant_id: uuid.UUID, scope: AccessScope = UNRESTRICTED) -> dict[str, Any]:
        rows = (
            await self.db.execute(
                select(Document.status, func.count(), func.coalesce(func.sum(Document.chunk_count), 0))
                .where(Document.tenant_id == tenant_id, self._visible(scope))
                .group_by(Document.status)
            )
        ).all()
        counts = {status: int(n) for status, n, _ in rows}
        return {
            "total_documents": sum(counts.values()),
            "processed_documents": counts.get(DocumentStatus.COMPLETED, 0),
            "processing_documents": counts.get(DocumentStatus.PROCESSING, 0),
            "pending_documents": counts.get(DocumentStatus.PENDING, 0),
            "failed_documents": counts.get(DocumentStatus.FAILED, 0),
            "total_chunks": int(sum(c for _, _, c in rows)),
        }
