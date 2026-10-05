"""Dataset lifecycle and question answering for Chat with CSV."""

from __future__ import annotations

import asyncio
import time
import uuid
from functools import partial
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.access import can_access, normalize_groups
from app.core.config import Settings
from app.core.errors import AppError, ConflictError, NotFoundError
from app.core.logging import get_logger
from app.core.metrics import DATASET_QUERIES
from app.datasets.planner import LLMPlanner, Plan, PlanError, RulePlanner, chart_for, summarize
from app.datasets.sql_safety import QueryResult, run_sql
from app.datasets.store import DatasetStorage, profile_csv
from app.llm.client import LLMClient
from app.models.dataset import Dataset
from app.services.audit import record_audit

logger = get_logger(__name__)


class DatasetService:
    def __init__(self, db: AsyncSession, settings: Settings, llm: LLMClient | None = None) -> None:
        self.db = db
        self.settings = settings
        self.llm = llm
        self.storage = DatasetStorage(settings.upload_dir)

    async def upload(self, tenant_id: uuid.UUID, user_id: uuid.UUID, filename: str, data: bytes,
                     access_groups: list[str] | None = None) -> Dataset:
        profiled = await asyncio.to_thread(profile_csv, filename, data, self.settings.max_upload_bytes,
                                           self.settings.dataset_max_rows)
        existing = (await self.db.execute(select(Dataset).where(Dataset.tenant_id == tenant_id,
                                                                Dataset.checksum == profiled.checksum))).scalar_one_or_none()
        if existing is not None:
            raise ConflictError(f"This file was already uploaded as dataset '{existing.name}'", code="DATASET_ALREADY_EXISTS")
        dataset_id = uuid.uuid4()
        path = await asyncio.to_thread(self.storage.save, str(tenant_id), str(dataset_id), profiled.parquet_bytes)
        dataset = Dataset(id=dataset_id, tenant_id=tenant_id, uploaded_by=user_id, name=profiled.name,
                          filename=profiled.filename, checksum=profiled.checksum, storage_path=path,
                          size_bytes=len(data), row_count=profiled.row_count, columns=profiled.columns,
                          access_groups=normalize_groups(access_groups))
        self.db.add(dataset)
        record_audit(self.db, "dataset.upload", tenant_id=tenant_id, user_id=user_id, resource_type="dataset",
                     resource_id=str(dataset_id), details={"filename": profiled.filename, "rows": profiled.row_count})
        await self.db.commit()
        return dataset

    async def list(self, tenant_id: uuid.UUID, groups: tuple[str, ...], is_admin: bool) -> list[Dataset]:
        rows = (await self.db.execute(select(Dataset).where(Dataset.tenant_id == tenant_id)
                                      .order_by(Dataset.created_at.desc()))).scalars()
        return [d for d in rows if can_access(d.access_groups, groups, is_admin)]

    async def get(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID, groups: tuple[str, ...], is_admin: bool) -> Dataset:
        dataset = (await self.db.execute(select(Dataset).where(Dataset.id == dataset_id,
                                                               Dataset.tenant_id == tenant_id))).scalar_one_or_none()
        if dataset is None or not can_access(dataset.access_groups, groups, is_admin):
            raise NotFoundError("Dataset not found", code="DATASET_NOT_FOUND")
        return dataset

    async def delete(self, dataset: Dataset, user_id: uuid.UUID) -> None:
        await asyncio.to_thread(self.storage.delete, dataset.storage_path)
        record_audit(self.db, "dataset.delete", tenant_id=dataset.tenant_id, user_id=user_id, resource_type="dataset",
                     resource_id=str(dataset.id))
        await self.db.delete(dataset)
        await self.db.commit()

    def _run(self, dataset: Dataset, sql: str, limit: int | None = None) -> QueryResult:
        return run_sql(dataset.storage_path, sql, limit=limit or self.settings.dataset_result_limit,
                       timeout_seconds=self.settings.dataset_query_timeout_seconds)

    async def preview(self, dataset: Dataset, rows: int = 20) -> QueryResult:
        return await asyncio.to_thread(self._run, dataset, "SELECT * FROM data", rows)

    async def query(self, dataset: Dataset, question: str, sql: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        explanation, planner_name = "", "sql"
        plan: Plan | None = None
        if sql:  # power users can edit and re-run the SQL; it is validated like any generated SQL
            result = await asyncio.to_thread(self._run, dataset, sql)
            plan = Plan(sql, "User-provided SQL", "table")
        else:
            result = None
            if self.llm is not None:
                try:
                    planner = LLMPlanner(self.llm, dataset.columns, dataset.row_count)
                    plan, result = await planner.plan(question, partial(self._run, dataset))
                    planner_name = "llm"
                except AppError as exc:
                    logger.warning("llm_text2sql_failed_falling_back", extra={"error": exc.code})
            if result is None:
                distinct_cache: dict[str, list[str]] = {}

                def distinct_values(column: str) -> list[str]:
                    if column not in distinct_cache:
                        res = self._run(dataset, f'SELECT DISTINCT "{column}" FROM data WHERE "{column}" IS NOT NULL', 2000)
                        distinct_cache[column] = [str(r[0]) for r in res.rows]
                    return distinct_cache[column]

                try:
                    plan = await asyncio.to_thread(RulePlanner(dataset.columns, distinct_values, dataset.row_count).plan, question)
                except PlanError:
                    DATASET_QUERIES.labels(planner="rules", outcome="not_understood").inc()
                    raise
                result = await asyncio.to_thread(self._run, dataset, plan.sql)
                planner_name = "rules"
            explanation = plan.explanation if plan else ""
        assert plan is not None and result is not None
        if planner_name == "llm" and self.llm is not None:
            try:
                answer = await LLMPlanner(self.llm, dataset.columns, dataset.row_count).answer(question, result)
            except AppError:
                answer = summarize(question, plan, result, dataset.columns)
        else:
            answer = summarize(question, plan, result, dataset.columns)
        DATASET_QUERIES.labels(planner=planner_name, outcome="answered").inc()
        return {
            "question": question, "answer": answer, "sql": plan.sql, "explanation": explanation,
            "planner": planner_name, "columns": result.columns, "rows": result.rows, "row_count": len(result.rows),
            "truncated": result.truncated, "chart": chart_for(result, dataset.columns),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
