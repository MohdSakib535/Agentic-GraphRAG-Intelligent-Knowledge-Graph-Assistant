from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ORMModel


class ChatRequest(BaseModel):
    conversation_id: uuid.UUID | None = Field(
        default=None, description="Existing conversation id; omit to start a new conversation"
    )
    message: str = Field(min_length=1, max_length=4000)

    model_config = ConfigDict(json_schema_extra={"example": {"message": "Which projects use Kafka?"}})


class Citation(BaseModel):
    index: int
    chunk_id: str | None = None
    document_id: str | None = None
    source_filename: str | None = None
    page_number: int | None = None
    section: str | None = None
    snippet: str | None = None
    kind: str = "chunk"  # chunk | graph


class TraceStep(BaseModel):
    step: str
    status: str = "done"
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    sources: list[Citation]
    retrieval_strategy: str
    confidence: float
    intent: str | None = None
    entities: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    retry_count: int = 0
    graph_evidence: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    verification: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int
    token_usage: dict[str, int] = Field(default_factory=dict)
    cached: bool = False


class ConversationOut(ORMModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(ORMModel):
    id: uuid.UUID
    role: str
    content: str
    retrieval_strategy: str | None
    confidence: float | None
    latency_ms: int | None = None
    sources: list[Any]
    trace: dict[str, Any]
    created_at: datetime
