from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base, TimestampMixin, UUIDPrimaryKey


class Dataset(UUIDPrimaryKey, TimestampMixin, Base):
    """A tabular dataset (CSV) stored as Parquet and queried with DuckDB ("Chat with CSV")."""

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("tenant_id", "checksum", name="uq_datasets_tenant_checksum"),)

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[list[Any]] = mapped_column(default=list, nullable=False)  # [{name, type, ...stats}]
    access_groups: Mapped[list[Any]] = mapped_column(default=list, nullable=False)
