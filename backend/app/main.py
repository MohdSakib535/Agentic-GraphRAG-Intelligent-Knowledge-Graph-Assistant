"""FastAPI application entry point."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.api import auth, chat, connectors, datasets, documents, evaluation, graph_admin, health, search
from app.core.config import Settings, get_settings
from app.core.container import build_container, open_postgres_checkpointer
from app.core.errors import AppError, RateLimitExceeded, error_payload
from app.core.logging import configure_logging, get_logger, request_id_ctx, tenant_id_ctx, user_id_ctx
from app.core.metrics import HTTP_LATENCY, HTTP_REQUESTS
from app.core.telemetry import instrument_app, instrument_engine, setup_tracing
from app.db.neo4j import close_async_driver, close_sync_driver, ensure_schema, init_async_driver
from app.db.postgres import dispose_async_engine, init_async_engine
from app.db.redis import close_redis, init_redis

logger = get_logger("app")

DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    async with AsyncExitStack() as stack:
        instrument_engine(init_async_engine(settings), settings)
        stack.push_async_callback(dispose_async_engine)
        driver = init_async_driver(settings)
        stack.push_async_callback(close_async_driver)
        stack.callback(close_sync_driver)
        redis_client = init_redis(settings)
        stack.push_async_callback(close_redis)
        try:
            await ensure_schema(driver, settings)
        except Exception as exc:  # the API can start degraded; readiness reports it
            logger.error("neo4j_schema_init_failed", extra={"error": type(exc).__name__})
        checkpointer = None
        if not settings.database_url.startswith("sqlite"):
            try:
                checkpointer = await open_postgres_checkpointer(settings, stack)
            except Exception as exc:
                logger.error("checkpointer_init_failed_using_memory", extra={"error": type(exc).__name__})
        app.state.container = build_container(settings, driver, redis_client, checkpointer)
        logger.info("startup_complete", extra={"environment": settings.environment})
        yield
    logger.info("shutdown_complete")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Request id propagation, access logging and security headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if incoming and len(incoming) <= 64 and incoming.replace("-", "").isalnum() else uuid.uuid4().hex
        request_id_ctx.set(request_id)
        tenant_id_ctx.set(None)
        user_id_ctx.set(None)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if not request.url.path.startswith(DOCS_PATHS):
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
            response.headers.setdefault("Cache-Control", "no-store")
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        rl = getattr(request.state, "rate_limit", None)
        if rl:
            response.headers["X-RateLimit-Limit"] = str(rl["limit"])
            response.headers["X-RateLimit-Remaining"] = str(max(0, rl["remaining"]))
        elapsed = time.perf_counter() - started
        route = getattr(request.scope.get("route"), "path", None) or "unmatched"  # template, never raw ids
        if route != "/metrics":
            HTTP_REQUESTS.labels(method=request.method, route=route, status=str(response.status_code)).inc()
            HTTP_LATENCY.labels(method=request.method, route=route).observe(elapsed)
        logger.info("http_request", extra={"method": request.method, "path": request.url.path,
                                           "status": response.status_code, "duration_ms": int(elapsed * 1000)})
        return response


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        headers = {}
        if isinstance(exc, RateLimitExceeded):
            headers["Retry-After"] = str(exc.retry_after)
        if exc.status_code >= 500:
            logger.warning("app_error", extra={"code": exc.code, "status": exc.status_code})
        return JSONResponse(error_payload(exc.code, exc.message, request_id_ctx.get(), exc.details),
                            status_code=exc.status_code, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return JSONResponse(error_payload("VALIDATION_ERROR", "Request validation failed", request_id_ctx.get(), details),
                            status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 401: "AUTHENTICATION_FAILED", 403: "FORBIDDEN"}.get(
            exc.status_code, "HTTP_ERROR")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(error_payload(code, message, request_id_ctx.get()), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error")  # full trace goes to logs only - never to the client
        return JSONResponse(error_payload("INTERNAL_ERROR", "An unexpected error occurred", request_id_ctx.get()),
                            status_code=500)


TAGS = [
    {"name": "Authentication", "description": "JWT authentication with rotating refresh tokens."},
    {"name": "Documents", "description": "Upload and manage PDF/DOCX/TXT/MD documents (async ingestion)."},
    {"name": "Chat", "description": "Agentic GraphRAG question answering (JSON or SSE streaming)."},
    {"name": "Search & Graph", "description": "Direct vector/graph/hybrid retrieval and graph exploration."},
    {"name": "Chat with CSV", "description": "Upload CSV datasets and ask analytical questions (validated SQL)."},
    {"name": "Connectors", "description": "Sync external sources (Google Drive) into the knowledge base (admin)."},
    {"name": "Evaluation", "description": "Benchmark Vector RAG vs GraphRAG vs Agentic GraphRAG."},
    {"name": "Health", "description": "Probes and public configuration."},
]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    setup_tracing(settings)
    app = FastAPI(
        title="Agentic GraphRAG API",
        description=(
            "Enterprise knowledge assistant combining a Neo4j knowledge graph, vector search and a LangGraph agent "
            "that chooses between vector, graph and hybrid retrieval, grades evidence, rewrites queries and "
            "verifies grounded answers with citations. All data is tenant-isolated."
        ),
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=TAGS,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,  # bearer tokens, no cookies
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
        max_age=600,
    )
    register_exception_handlers(app)
    instrument_app(app, settings)
    for router in (health.router, auth.router, documents.router, chat.router, search.router, datasets.router,
                   graph_admin.router, connectors.router, evaluation.router):
        app.include_router(router, prefix=settings.api_prefix)
    app.include_router(health.metrics_router)
    return app


app = create_app()
