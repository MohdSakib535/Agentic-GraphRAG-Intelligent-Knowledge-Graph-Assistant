from __future__ import annotations

import enum
import uuid
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base, TimestampMixin, UUIDPrimaryKey


class DocumentStatus(enum.StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Document(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "checksum", name="uq_documents_tenant_checksum"),
        Index("ix_documents_tenant_status", "tenant_id", "status"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=DocumentStatus.PENDING, nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    entity_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    relationship_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict, nullable=False)
    # Document-level permissions: empty = visible to the whole tenant; otherwise only to members of
    # at least one listed group (admins always see everything).
    access_groups: Mapped[list[Any]] = mapped_column(default=list, nullable=False)
    source: Mapped[str] = mapped_column(String(20), default="upload", nullable=False)  # upload | google_drive
    connector_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("connectors.id", ondelete="SET NULL"))
    external_id: Mapped[str | None] = mapped_column(String(200), index=True)
    external_modified_at: Mapped[str | None] = mapped_column(String(40))
