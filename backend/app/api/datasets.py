"""Chat with CSV: upload tabular data and ask analytical questions (answered with validated SQL)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.core.dependencies import ContainerDep, CurrentUserDep, DBSession, RateLimit, SettingsDep
from app.core.errors import ValidationFailed
from app.datasets.service import DatasetService
from app.ingestion.loader import read_limited
from app.schemas.common import ERROR_RESPONSES
from app.schemas.dataset import DatasetDetail, DatasetOut, DatasetQueryRequest, DatasetQueryResponse

router = APIRouter(prefix="/datasets", tags=["Chat with CSV"], responses=ERROR_RESPONSES)


@router.post("/upload", response_model=DatasetOut, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(RateLimit("upload"))], summary="Upload a CSV dataset")
async def upload_dataset(
    user: CurrentUserDep, db: DBSession, settings: SettingsDep, container: ContainerDep,
    file: Annotated[UploadFile, File(description="CSV/TSV with a header row")],
    access_groups: Annotated[str | None, Form(description="Comma-separated groups (empty = whole tenant)")] = None,
) -> DatasetOut:
    """Types are inferred, columns profiled (nulls, distinct values, ranges) and the data stored as Parquet."""
    groups = [g.strip() for g in (access_groups or "").split(",") if g.strip()]
    if groups and not user.is_admin and not set(groups) <= set(user.groups):
        raise ValidationFailed("You can only restrict datasets to groups you belong to", code="INVALID_ACCESS_GROUPS")
    data = read_limited(file.file, settings.max_upload_bytes)
    dataset = await DatasetService(db, settings, container.llm).upload(user.tenant_id, user.id, file.filename or "data.csv",
                                                                       data, groups)
    return DatasetOut.model_validate(dataset)


@router.get("", response_model=list[DatasetOut], summary="List datasets I can access")
async def list_datasets(user: CurrentUserDep, db: DBSession, settings: SettingsDep) -> list[DatasetOut]:
    items = await DatasetService(db, settings).list(user.tenant_id, user.groups, user.is_admin)
    return [DatasetOut.model_validate(d) for d in items]


@router.get("/{dataset_id}", response_model=DatasetDetail, summary="Dataset schema, profile and preview rows")
async def get_dataset(dataset_id: uuid.UUID, user: CurrentUserDep, db: DBSession, settings: SettingsDep) -> DatasetDetail:
    service = DatasetService(db, settings)
    dataset = await service.get(user.tenant_id, dataset_id, user.groups, user.is_admin)
    preview = await service.preview(dataset)
    detail = DatasetDetail.model_validate(dataset)
    detail.preview_columns, detail.preview_rows = preview.columns, preview.rows
    return detail


@router.post("/{dataset_id}/query", response_model=DatasetQueryResponse, dependencies=[Depends(RateLimit("chat"))],
             summary="Ask a question about a dataset")
async def query_dataset(dataset_id: uuid.UUID, body: DatasetQueryRequest, user: CurrentUserDep, db: DBSession,
                        settings: SettingsDep, container: ContainerDep) -> DatasetQueryResponse:
    """The question becomes a single read-only SQL query (LLM text-to-SQL, or the offline planner), which is
    validated against DuckDB's parse tree and run in a sandbox. Returns the answer, the SQL, the rows and a
    chart suggestion."""
    service = DatasetService(db, settings, container.llm)
    dataset = await service.get(user.tenant_id, dataset_id, user.groups, user.is_admin)
    return DatasetQueryResponse(**await service.query(dataset, body.question, body.sql))


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a dataset")
async def delete_dataset(dataset_id: uuid.UUID, user: CurrentUserDep, db: DBSession, settings: SettingsDep) -> None:
    service = DatasetService(db, settings)
    await service.delete(await service.get(user.tenant_id, dataset_id, user.groups, user.is_admin), user.id)
