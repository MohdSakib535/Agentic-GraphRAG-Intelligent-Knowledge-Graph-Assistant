from __future__ import annotations

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base, TimestampMixin, UUIDPrimaryKey


class Tenant(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "tenants"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(220), unique=True, index=True, nullable=False)
    # Google Workspace domain whose SSO users join this tenant automatically (e.g. "techcorp.com").
    sso_domain: Mapped[str | None] = mapped_column(String(253), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
