"""Shared plumbing for agent tools: tenant isolation, timeouts, structured errors, logging.

The tenant id and the document access scope are taken from the *runnable config*
(set by the server from the authenticated user), never from tool arguments - a model
cannot choose a tenant or widen its permissions.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.core.access import AccessScope, use_scope
from app.core.errors import AppError, ValidationFailed
from app.core.logging import get_logger
from app.utils.ids import is_uuid

logger = get_logger("app.agents.tools")


class ToolOutput(BaseModel):
    ok: bool
    tool: str
    latency_ms: int
    data: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, str] | None = None


def tenant_from_config(config: RunnableConfig | None) -> str:
    tenant_id = ((config or {}).get("configurable") or {}).get("tenant_id")
    if not tenant_id or not is_uuid(str(tenant_id)):
        raise ValidationFailed("Tool invoked without an authenticated tenant context")
    return str(tenant_id)


async def run_tool(
    name: str, config: RunnableConfig | None, timeout: float, fn: Callable[[str], Awaitable[dict[str, Any]]]
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        tenant_id = tenant_from_config(config)
        scope = AccessScope.from_config(((config or {}).get("configurable") or {}).get("denied_document_ids"))
        with use_scope(scope):
            data = await asyncio.wait_for(fn(tenant_id), timeout=timeout)
        out = ToolOutput(ok=True, tool=name, latency_ms=int((time.perf_counter() - started) * 1000), data=data)
    except TimeoutError:
        out = ToolOutput(ok=False, tool=name, latency_ms=int((time.perf_counter() - started) * 1000),
                         error={"code": "TOOL_TIMEOUT", "message": f"{name} timed out"})
    except AppError as exc:
        out = ToolOutput(ok=False, tool=name, latency_ms=int((time.perf_counter() - started) * 1000),
                         error={"code": exc.code, "message": exc.message})
    except Exception as exc:
        logger.exception("tool_failed", extra={"tool": name})
        out = ToolOutput(ok=False, tool=name, latency_ms=int((time.perf_counter() - started) * 1000),
                         error={"code": "TOOL_FAILED", "message": type(exc).__name__})
    logger.info("tool_executed", extra={"tool": name, "ok": out.ok, "latency_ms": out.latency_ms,
                                        "error_code": out.error["code"] if out.error else None})
    return out.model_dump()
