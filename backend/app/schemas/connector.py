from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class ConnectorCreate(BaseModel):
    type: Literal["google_drive"] = "google_drive"
    name: str = Field(min_length=1, max_length=200)
    folder_id: str = Field(min_length=5, max_length=200, pattern=r"^[A-Za-z0-9_\-]+$",
                           description="Drive folder id (the last part of the folder URL)")
    service_account_json: str = Field(min_length=20, max_length=20000,
                                      description="Service-account key JSON; stored encrypted, never returned")
    access_groups: list[str] = Field(default_factory=list, max_length=20)
    sync_now: bool = True


class ConnectorOut(ORMModel):
    id: uuid.UUID
    type: str
    name: str
    config: dict[str, Any]
    access_groups: list[str]
    status: str
    last_sync_at: datetime | None
    last_error: str | None
    last_sync_stats: dict[str, Any]
    created_at: datetime
