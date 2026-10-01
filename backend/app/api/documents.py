"""Document upload and management. Processing happens asynchronously in Celery."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from app.core.dependencies import ContainerDep, CurrentUserDep, DBSession, RateLimit, SettingsDep
from app.ingestion.loader import read_limited, validate_upload
from app.schemas.common import ERROR_RESPONSES, Page
from app.schemas.document import DocumentDetail, DocumentOut, DocumentStats, DocumentStatusOut, JobOut, UploadResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"], responses=ERROR_RESPONSES)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(RateLimit("upload"))], summary="Upload a document for ingestion")
async def upload_document(
    user: CurrentUserDep, db: DBSession, settings: SettingsDep, container: ContainerDep,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or MD file")],
) -> UploadResponse:
    """Validates and stores the file, records metadata, creates an ingestion job and queues it.

    Returns **202 Accepted** immediately; poll `GET /documents/{id}/status` for progress.
    """
    data = read_limited(file.file, settings.max_upload_bytes)
    validated = validate_upload(file.filename, file.content_type, data, settings.max_upload_bytes)
    document, job = await DocumentService(db, settings, container.cache).upload(user.tenant_id, user.id, validated)
    return UploadResponse(document=DocumentOut.model_validate(document), job=JobOut.model_validate(job))


@router.get("", response_model=Page[DocumentOut], summary="List documents")
async def list_documents(
    user: CurrentUserDep, db: DBSession, settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50, offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[str | None, Query(alias="status", pattern="(?i)^(pending|processing|completed|failed)$")] = None,
) -> Page[DocumentOut]:
    items, total = await DocumentService(db, settings).list(user.tenant_id, limit, offset, status_filter)
    return Page(items=[DocumentOut.model_validate(d) for d in items], total=total, limit=limit, offset=offset)


@router.get("/stats", response_model=DocumentStats, summary="Document counts by status")
async def document_stats(user: CurrentUserDep, db: DBSession, settings: SettingsDep) -> DocumentStats:
    return DocumentStats(**await DocumentService(db, settings).stats(user.tenant_id))


@router.get("/{document_id}", response_model=DocumentDetail, summary="Get document metadata")
async def get_document(document_id: uuid.UUID, user: CurrentUserDep, db: DBSession, settings: SettingsDep) -> DocumentDetail:
    service = DocumentService(db, settings)
    document = await service.get(user.tenant_id, document_id)
    job = await service.latest_job(user.tenant_id, document_id)
    detail = DocumentDetail.model_validate(document)
    detail.latest_job = JobOut.model_validate(job) if job else None
    return detail


@router.get("/{document_id}/status", response_model=DocumentStatusOut, summary="Ingestion status and progress")
async def document_status(document_id: uuid.UUID, user: CurrentUserDep, db: DBSession, settings: SettingsDep) -> DocumentStatusOut:
    service = DocumentService(db, settings)
    document = await service.get(user.tenant_id, document_id)
    job = await service.latest_job(user.tenant_id, document_id)
    return DocumentStatusOut(document_id=document.id, status=document.status,
                             job=JobOut.model_validate(job) if job else None)


@router.post("/{document_id}/reprocess", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(RateLimit("upload"))], summary="Re-run ingestion for a document")
async def reprocess_document(document_id: uuid.UUID, user: CurrentUserDep, db: DBSession, settings: SettingsDep) -> JobOut:
    job = await DocumentService(db, settings).reprocess(user.tenant_id, user.id, document_id)
    return JobOut.model_validate(job)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document")
async def delete_document(document_id: uuid.UUID, user: CurrentUserDep, db: DBSession, settings: SettingsDep,
                          container: ContainerDep) -> None:
    """Deletes the document, its chunks and embeddings, and prunes graph entities/relationships
    that were only supported by this document."""
    await DocumentService(db, settings, container.cache).delete(user.tenant_id, user.id, document_id)
