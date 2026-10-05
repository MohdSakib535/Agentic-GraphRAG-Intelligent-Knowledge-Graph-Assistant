"""Liveness/readiness probes and public (non-secret) configuration."""

from __future__ import annotations

import asyncio
import hmac
from typing import Any

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import func, select

from app.core import metrics
from app.core.dependencies import CurrentUserDep, DBSession, SettingsDep
from app.core.errors import AuthenticationError, NotFoundError
from app.db.neo4j import ping_neo4j
from app.db.postgres import ping_postgres
from app.db.redis import ping_redis
from app.models.document import Document, DocumentStatus
from app.models.job import IngestionJob, JobStatus

router = APIRouter(tags=["Health"])
metrics_router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _check(name: str, coro: Any) -> tuple[str, dict[str, Any]]:
    try:
        await asyncio.wait_for(coro, timeout=3.0)
        return name, {"status": "ok"}
    except Exception as exc:
        return name, {"status": "unavailable", "error": type(exc).__name__}


@router.get("/health/ready", summary="Readiness probe (PostgreSQL, Neo4j, Redis)")
async def ready(response: Response) -> dict[str, Any]:
    checks = dict(await asyncio.gather(
        _check("postgres", ping_postgres()), _check("neo4j", ping_neo4j()), _check("redis", ping_redis())
    ))
    healthy = all(c["status"] == "ok" for c in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if healthy else "degraded", "checks": checks}


@router.get("/settings", summary="Effective non-secret configuration")
async def public_settings(user: CurrentUserDep, settings: SettingsDep) -> dict[str, Any]:
    return settings.public_dict()


@metrics_router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request, settings: SettingsDep, db: DBSession) -> Response:
    """Prometheus exposition. Labels are bounded enums; no tenant ids, users or content."""
    if not settings.metrics_enabled:
        raise NotFoundError()
    token = settings.metrics_token.get_secret_value()
    if token and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
        raise AuthenticationError("Metrics token required")
    for model, states, gauge in ((IngestionJob, JobStatus, metrics.INGESTION_JOBS),
                                 (Document, DocumentStatus, metrics.DOCUMENTS)):
        counts = dict((await db.execute(select(model.status, func.count()).group_by(model.status))).all())
        for state in states:  # every known status, so finished states drop back to 0
            gauge.labels(status=state.value).set(counts.get(state.value, 0))
    body, content_type = metrics.render()
    return Response(body, media_type=content_type)
