"""Liveness/readiness probes and public (non-secret) configuration."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Response, status

from app.core.dependencies import CurrentUserDep, SettingsDep
from app.db.neo4j import ping_neo4j
from app.db.postgres import ping_postgres
from app.db.redis import ping_redis

router = APIRouter(tags=["Health"])


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
