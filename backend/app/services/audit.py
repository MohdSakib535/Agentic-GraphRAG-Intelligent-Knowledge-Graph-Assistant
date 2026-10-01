"""Audit trail for security-relevant actions (never stores secrets or document content)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import redact, request_id_ctx
from app.models.audit import AuditLog


def record_audit(
    db: AsyncSession,
    action: str,
    *,
    tenant_id: uuid.UUID | None,
    user_id: uuid.UUID | None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    db.add(
        AuditLog(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=request_id_ctx.get(),
            details=redact(details or {}),
        )
    )
