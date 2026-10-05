from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base, TimestampMixin, UUIDPrimaryKey


class Connector(UUIDPrimaryKey, TimestampMixin, Base):
    """An external content source (e.g. a Google Drive folder) synced into the knowledge base."""

    __tablename__ = "connectors"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    type: Mapped[str] = mapped_column(String(30), nullable=False)  # google_drive
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    # Fernet-encrypted credentials (service-account JSON). Never returned by the API.
    encrypted_credentials: Mapped[str] = mapped_column(Text, nullable=False)
    access_groups: Mapped[list[Any]] = mapped_column(default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="IDLE", nullable=False)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    last_sync_stats: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
