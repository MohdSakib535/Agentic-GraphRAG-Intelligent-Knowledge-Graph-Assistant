from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base, TimestampMixin, UUIDPrimaryKey


class EvaluationRun(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "evaluation_runs"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(20), default="QUEUED", nullable=False)
    systems: Mapped[list[Any]] = mapped_column(default=list, nullable=False)
    question_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvaluationResult(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "evaluation_results"

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    system: Mapped[str] = mapped_column(String(40), nullable=False)
    question_id: Mapped[str] = mapped_column(String(20), nullable=False)
    category: Mapped[str] = mapped_column(String(40), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    expected_strategy: Mapped[str | None] = mapped_column(String(20))
    selected_strategy: Mapped[str | None] = mapped_column(String(20))
    correctness: Mapped[float] = mapped_column(Float, default=0.0)
    faithfulness: Mapped[float] = mapped_column(Float, default=0.0)
    context_relevance: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_recall: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    token_usage: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
