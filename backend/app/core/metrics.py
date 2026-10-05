"""Prometheus metrics (exposed at ``/metrics``).

Label values are always bounded enums (strategy, route template, status, cache namespace) - never
tenant ids, user ids or question text - so cardinality stays small and nothing sensitive is exported.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_REQUESTS = Counter("graphrag_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "graphrag_http_request_duration_seconds", "HTTP request latency", ["method", "route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
CHAT_TURNS = Counter("graphrag_chat_turns_total", "Agent turns", ["strategy", "outcome"])
CHAT_LATENCY = Histogram(
    "graphrag_chat_turn_duration_seconds", "End-to-end agent turn latency", ["strategy"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60),
)
AGENT_RETRIES = Counter("graphrag_agent_query_rewrites_total", "Query rewrites performed by the agent")
LLM_TOKENS = Counter("graphrag_llm_tokens_total", "LLM tokens consumed", ["kind"])
CACHE_EVENTS = Counter("graphrag_cache_events_total", "Cache lookups", ["namespace", "result"])
INGESTION_JOBS = Counter("graphrag_ingestion_jobs_total", "Finished ingestion jobs", ["status"])
DATASET_QUERIES = Counter("graphrag_dataset_queries_total", "Chat-with-CSV queries", ["planner", "outcome"])
FEEDBACK = Counter("graphrag_feedback_total", "Answer feedback", ["rating"])


def record_cache(namespace: str, hit: bool) -> None:
    CACHE_EVENTS.labels(namespace=namespace, result="hit" if hit else "miss").inc()


def record_chat_turn(strategy: str | None, outcome: str, seconds: float, retries: int, usage: dict[str, int]) -> None:
    label = strategy or "UNKNOWN"
    CHAT_TURNS.labels(strategy=label, outcome=outcome).inc()
    CHAT_LATENCY.labels(strategy=label).observe(seconds)
    if retries:
        AGENT_RETRIES.inc(retries)
    LLM_TOKENS.labels(kind="prompt").inc(usage.get("prompt_tokens", 0))
    LLM_TOKENS.labels(kind="completion").inc(usage.get("completion_tokens", 0))


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
