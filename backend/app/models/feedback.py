from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, SmallInteger, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base, TimestampMixin, UUIDPrimaryKey


class MessageFeedback(UUIDPrimaryKey, TimestampMixin, Base):
    """A user's thumbs-up (+1) / thumbs-down (-1) rating of an assistant message."""

    __tablename__ = "message_feedback"
    __table_args__ = (UniqueConstraint("message_id", "user_id", name="uq_message_feedback_message_user"),)

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    comment: Mapped[str | None] = mapped_column(String(1000))
