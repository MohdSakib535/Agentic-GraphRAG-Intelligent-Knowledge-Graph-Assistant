from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import ORMModel


class JobOut(ORMModel):
    id: uuid.UUID
    document_id: uuid.UUID
    status: str
    stage: str
    progress: int
    attempts: int
    error_code: str | None
    error_message: str | None
    stats: dict[str, Any]
    stage_history: list[Any]
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class DocumentOut(ORMModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    filename: str
    file_type: str
    size_bytes: int
    checksum: str
    status: str
    title: str | None
    page_count: int | None
    chunk_count: int
    entity_count: int
    relationship_count: int
    error_message: str | None
    metadata: dict[str, Any] = Field(validation_alias="metadata_")
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentOut):
    latest_job: JobOut | None = None


class DocumentStatusOut(ORMModel):
    document_id: uuid.UUID
    status: str
    job: JobOut | None


class UploadResponse(ORMModel):
    document: DocumentOut
    job: JobOut


class DocumentStats(ORMModel):
    total_documents: int
    processed_documents: int
    processing_documents: int
    failed_documents: int
    pending_documents: int
    total_chunks: int
