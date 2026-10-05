from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ORMModel


class DatasetOut(ORMModel):
    id: uuid.UUID
    name: str
    filename: str
    size_bytes: int
    row_count: int
    columns: list[dict[str, Any]]
    access_groups: list[str]
    created_at: datetime


class DatasetDetail(DatasetOut):
    preview_columns: list[str] = Field(default_factory=list)
    preview_rows: list[list[Any]] = Field(default_factory=list)


class DatasetQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    sql: str | None = Field(default=None, max_length=5000,
                            description="Optional SQL to run instead (validated; only SELECT over table `data`)")

    model_config = ConfigDict(json_schema_extra={"example": {"question": "Which department has the highest average salary?"}})


class DatasetQueryResponse(BaseModel):
    question: str
    answer: str
    sql: str
    explanation: str
    planner: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    chart: dict[str, Any] | None
    latency_ms: int
