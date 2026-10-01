from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base, TimestampMixin, UUIDPrimaryKey


class JobStatus(enum.StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobStage(enum.StrEnum):
    QUEUED = "QUEUED"
    PARSING = "PARSING"
    CLEANING = "CLEANING"
    CHUNKING = "CHUNKING"
    EXTRACTING_ENTITIES = "EXTRACTING_ENTITIES"
    EXTRACTING_RELATIONSHIPS = "EXTRACTING_RELATIONSHIPS"
    RESOLVING_ENTITIES = "RESOLVING_ENTITIES"
    EMBEDDING = "EMBEDDING"
    BUILDING_GRAPH = "BUILDING_GRAPH"
    INDEXING = "INDEXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


STAGE_PROGRESS: dict[str, int] = {
    JobStage.QUEUED: 0,
    JobStage.PARSING: 10,
    JobStage.CLEANING: 15,
    JobStage.CHUNKING: 20,
    JobStage.EXTRACTING_ENTITIES: 30,
    JobStage.EXTRACTING_RELATIONSHIPS: 50,
    JobStage.RESOLVING_ENTITIES: 60,
    JobStage.EMBEDDING: 70,
    JobStage.BUILDING_GRAPH: 85,
    JobStage.INDEXING: 95,
    JobStage.COMPLETED: 100,
}


class IngestionJob(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.QUEUED, nullable=False)
    stage: Mapped[str] = mapped_column(String(40), default=JobStage.QUEUED, nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    stage_history: Mapped[list[Any]] = mapped_column(default=list, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
