"""OpenTelemetry tracing (opt-in via ``OTEL_ENABLED``).

Auto-instruments FastAPI, SQLAlchemy, Redis, httpx (OpenAI / Google calls) and Celery, and the agent adds
one span per LangGraph node. Spans carry only bounded, non-sensitive attributes: never question text,
document content, tokens or keys. Without ``OTEL_ENABLED`` every helper here is a cheap no-op.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_tracer = trace.get_tracer("agentic-graphrag")
_configured = False


def setup_tracing(settings: Settings, *, component: str = "api") -> bool:
    """Install the global tracer provider and library instrumentations once per process."""
    global _configured
    if not settings.otel_enabled or _configured:
        return _configured
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({
        "service.name": settings.otel_service_name if component == "api" else f"{settings.otel_service_name}-{component}",
        "deployment.environment": settings.environment,
    })
    provider = TracerProvider(resource=resource)
    endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces"
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    HTTPXClientInstrumentor().instrument()
    RedisInstrumentor().instrument()
    if component == "worker":
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        CeleryInstrumentor().instrument()
    _configured = True
    logger.info("tracing_enabled", extra={"component": component})
    return True


def instrument_app(app: Any, settings: Settings) -> None:
    if not settings.otel_enabled:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics", exclude_spans=["receive", "send"])


def instrument_engine(engine: Any, settings: Settings) -> None:
    if not settings.otel_enabled or engine is None:
        return
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    SQLAlchemyInstrumentor().instrument(engine=getattr(engine, "sync_engine", engine), enable_commenter=False)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Manual span; attribute values must be bounded labels (strategy, node, counts)."""
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current
