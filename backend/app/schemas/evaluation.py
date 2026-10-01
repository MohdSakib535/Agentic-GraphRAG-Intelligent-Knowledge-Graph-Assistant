from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel

SystemName = Literal["vector_rag", "graph_rag", "agentic_graphrag"]


class EvaluationRunRequest(BaseModel):
    systems: list[SystemName] = Field(default_factory=lambda: ["vector_rag", "graph_rag", "agentic_graphrag"])
    categories: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=200)


class EvaluationRunOut(ORMModel):
    id: uuid.UUID
    status: str
    systems: list[Any]
    question_count: int
    summary: dict[str, Any]
    error_message: str | None
    created_at: datetime
    finished_at: datetime | None


class EvaluationResultOut(ORMModel):
    id: uuid.UUID
    system: str
    question_id: str
    category: str
    question: str
    answer: str
    expected_strategy: str | None
    selected_strategy: str | None
    correctness: float
    faithfulness: float
    context_relevance: float
    retrieval_recall: float
    latency_ms: int
    token_usage: int


class EvaluationResultsResponse(BaseModel):
    run: EvaluationRunOut | None
    results: list[EvaluationResultOut]
