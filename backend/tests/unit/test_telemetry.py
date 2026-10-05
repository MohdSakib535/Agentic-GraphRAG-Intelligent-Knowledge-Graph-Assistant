"""Agent tracing emits one span per LangGraph node, with no question text or tenant data."""

from __future__ import annotations

from agent_helpers import ask
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.config import Settings
from app.core.telemetry import setup_tracing

QUESTION = "Who manages Project Alpha?"


async def test_agent_nodes_are_traced_without_sensitive_attributes(container) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    await ask(container, QUESTION)
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {"agent.analyze_query", "agent.generate_answer"} <= names, names
    for s in spans:
        values = " ".join(str(v) for v in (s.attributes or {}).values())
        assert "Project Alpha" not in values and "Who manages" not in values


def test_tracing_is_opt_in() -> None:
    assert setup_tracing(Settings(otel_enabled=False)) is False
