"""Content connectors (admin): Google Drive folder sync."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select

from app.connectors.google_drive import GoogleDriveClient, parse_service_account
from app.core.access import normalize_groups
from app.core.crypto import encrypt
from app.core.dependencies import ContainerDep, CurrentUser, DBSession, SettingsDep, require_admin
from app.core.errors import NotFoundError, RedisUnavailable
from app.models.connector import Connector
from app.models.document import Document
from app.schemas.common import ERROR_RESPONSES
from app.schemas.connector import ConnectorCreate, ConnectorOut
from app.services.audit import record_audit
from app.services.document_service import DocumentService

router = APIRouter(prefix="/connectors", tags=["Connectors"], responses=ERROR_RESPONSES)
AdminDep = Annotated[CurrentUser, Depends(require_admin)]


async def _get(db: DBSession, tenant_id: uuid.UUID, connector_id: uuid.UUID) -> Connector:
    connector = (await db.execute(select(Connector).where(Connector.id == connector_id,
                                                          Connector.tenant_id == tenant_id))).scalar_one_or_none()
    if connector is None:
        raise NotFoundError("Connector not found", code="CONNECTOR_NOT_FOUND")
    return connector


async def _enqueue_sync(connector: Connector) -> None:
    from app.workers.tasks import sync_connector

    try:
        await asyncio.to_thread(sync_connector.apply_async, args=[str(connector.id)])
    except Exception as exc:
        raise RedisUnavailable("The task queue is unavailable") from exc


@router.post("", response_model=ConnectorOut, status_code=status.HTTP_201_CREATED, summary="Connect a Google Drive folder")
async def create_connector(body: ConnectorCreate, admin: AdminDep, db: DBSession, settings: SettingsDep) -> ConnectorOut:
    """Validates the service-account key and that the folder is shared with it, stores the key encrypted,
    and (by default) starts the first sync."""
    info = parse_service_account(body.service_account_json)
    client = GoogleDriveClient(settings, info)
    try:
        folder = await asyncio.to_thread(client.check_access, body.folder_id)
    finally:
        client.close()
    connector = Connector(tenant_id=admin.tenant_id, created_by=admin.id, type=body.type, name=body.name,
                          config={"folder_id": body.folder_id, "folder_name": folder.get("name"),
                                  "service_account": info["client_email"]},
                          encrypted_credentials=encrypt(settings, body.service_account_json),
                          access_groups=normalize_groups(body.access_groups))
    db.add(connector)
    record_audit(db, "connector.create", tenant_id=admin.tenant_id, user_id=admin.id, resource_type="connector",
                 details={"type": body.type, "folder_id": body.folder_id})
    await db.commit()
    if body.sync_now:
        await _enqueue_sync(connector)
    await db.refresh(connector)
    return ConnectorOut.model_validate(connector)


@router.get("", response_model=list[ConnectorOut], summary="List connectors")
async def list_connectors(admin: AdminDep, db: DBSession) -> list[ConnectorOut]:
    rows = await db.execute(select(Connector).where(Connector.tenant_id == admin.tenant_id).order_by(Connector.created_at))
    return [ConnectorOut.model_validate(c) for c in rows.scalars()]


@router.get("/{connector_id}", response_model=ConnectorOut, summary="Connector status and last sync statistics")
async def get_connector(connector_id: uuid.UUID, admin: AdminDep, db: DBSession) -> ConnectorOut:
    return ConnectorOut.model_validate(await _get(db, admin.tenant_id, connector_id))


@router.post("/{connector_id}/sync", response_model=ConnectorOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Run an incremental sync now")
async def sync_connector(connector_id: uuid.UUID, admin: AdminDep, db: DBSession) -> ConnectorOut:
    connector = await _get(db, admin.tenant_id, connector_id)
    await _enqueue_sync(connector)
    await db.refresh(connector)
    return ConnectorOut.model_validate(connector)


@router.delete("/{connector_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Remove a connector")
async def delete_connector(connector_id: uuid.UUID, admin: AdminDep, db: DBSession, settings: SettingsDep,
                           container: ContainerDep,
                           delete_documents: Annotated[bool, Query(description="Also delete its documents")] = True) -> None:
    connector = await _get(db, admin.tenant_id, connector_id)
    if delete_documents:
        docs = (await db.execute(select(Document.id).where(Document.connector_id == connector.id))).scalars().all()
        service = DocumentService(db, settings, container.cache)
        for doc_id in docs:
            await service.delete(admin.tenant_id, admin.id, doc_id)
    await db.delete(connector)
    record_audit(db, "connector.delete", tenant_id=admin.tenant_id, user_id=admin.id, resource_type="connector",
                 resource_id=str(connector_id))
    await db.commit()
