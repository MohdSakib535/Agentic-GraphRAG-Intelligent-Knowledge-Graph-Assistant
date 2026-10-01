"""Evaluation endpoints: run the benchmark (async, Celery) and fetch results."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.core.dependencies import CurrentUserDep, DBSession
from app.core.errors import RedisUnavailable
from app.models.evaluation import EvaluationRun
from app.schemas.common import ERROR_RESPONSES
from app.schemas.evaluation import (
    EvaluationResultOut,
    EvaluationResultsResponse,
    EvaluationRunOut,
    EvaluationRunRequest,
)
from app.services.evaluation_service import latest_results, load_dataset

router = APIRouter(prefix="/evaluation", tags=["Evaluation"], responses=ERROR_RESPONSES)


@router.post("/run", response_model=EvaluationRunOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Start an evaluation run (Vector RAG vs GraphRAG vs Agentic GraphRAG)")
async def run_evaluation(body: EvaluationRunRequest, user: CurrentUserDep, db: DBSession) -> EvaluationRunOut:
    from app.workers.tasks import run_evaluation as task

    run = EvaluationRun(tenant_id=user.tenant_id, created_by=user.id, status="QUEUED", systems=list(body.systems),
                        summary={"options": {"categories": body.categories, "limit": body.limit}})
    db.add(run)
    await db.commit()
    try:
        await asyncio.to_thread(task.apply_async, args=[str(run.id)], task_id=str(run.id))
    except Exception as exc:
        run.status, run.error_message = "FAILED", "queue unavailable"
        await db.commit()
        raise RedisUnavailable("The task queue is unavailable") from exc
    return EvaluationRunOut.model_validate(run)


@router.get("/results", response_model=EvaluationResultsResponse, summary="Latest (or a specific) run's results")
async def evaluation_results(user: CurrentUserDep, db: DBSession,
                             run_id: Annotated[uuid.UUID | None, Query()] = None) -> EvaluationResultsResponse:
    run, rows = await latest_results(db, user.tenant_id, run_id)
    return EvaluationResultsResponse(
        run=EvaluationRunOut.model_validate(run) if run else None,
        results=[EvaluationResultOut.model_validate(r) for r in rows],
    )


@router.get("/dataset", summary="The evaluation question set")
async def evaluation_dataset(user: CurrentUserDep) -> dict[str, object]:
    questions = load_dataset()
    return {"total": len(questions), "questions": questions}
