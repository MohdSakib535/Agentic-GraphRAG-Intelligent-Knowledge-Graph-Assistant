"""OpenAI-compatible chat model wrapper.

* Structured output is requested via JSON schema and then **re-validated** with
  Pydantic - LLM JSON is never trusted blindly.
* Every call has a timeout and bounded retries.
* Token usage is accumulated into a request-scoped context variable so that the
  chat API and the evaluation harness can report it.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import LLMOutputError, LLMTimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_ms: float = 0.0
    by_task: dict[str, int] = field(default_factory=dict)

    def add(self, task: str, usage: dict[str, Any] | None, latency_ms: float) -> None:
        self.calls += 1
        self.latency_ms += latency_ms
        if usage:
            prompt = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            completion = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.by_task[task] = self.by_task.get(task, 0) + prompt + completion

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.calls,
            "llm_latency_ms": int(self.latency_ms),
        }


_usage_ctx: contextvars.ContextVar[UsageTracker | None] = contextvars.ContextVar("llm_usage", default=None)


def start_usage_tracking() -> UsageTracker:
    tracker = UsageTracker()
    _usage_ctx.set(tracker)
    return tracker


def current_usage() -> UsageTracker | None:
    return _usage_ctx.get()


def _record(task: str, message: Any, started: float) -> None:
    tracker = _usage_ctx.get()
    if tracker is None:
        return
    usage = getattr(message, "usage_metadata", None) if message is not None else None
    tracker.add(task, usage, (time.perf_counter() - started) * 1000)


def to_messages(system: str, user: str, history: Sequence[BaseMessage] = ()) -> list[BaseMessage]:
    return [SystemMessage(content=system), *history, HumanMessage(content=user)]


def _parse_structured(schema: type[TModel], raw: AIMessage | None, parsed: Any) -> TModel:
    if isinstance(parsed, schema):
        return schema.model_validate(parsed.model_dump())
    if isinstance(parsed, dict):
        return schema.model_validate(parsed)
    if raw is not None and isinstance(raw.content, str) and raw.content.strip():
        text = raw.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{") :]
        return schema.model_validate(json.loads(text))
    raise LLMOutputError("Structured output missing from model response")


class LLMClient:
    """Thin, testable wrapper around an OpenAI-compatible LangChain chat model."""

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self.settings = settings
        if model is None:
            from langchain_openai import ChatOpenAI

            api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
            model = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                api_key=api_key,
                base_url=settings.openai_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                stream_usage=True,
            )
        self.model = model
        self._semaphore: asyncio.Semaphore | None = None

    def _sem(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.llm_max_concurrency)
        return self._semaphore

    def _structured_runnable(self, schema: type[BaseModel]) -> Any:
        return self.model.with_structured_output(schema, method="json_schema", include_raw=True, strict=False)

    # ----------------------------------------------------------------- async
    async def astructured(self, schema: type[TModel], messages: list[BaseMessage], *, task: str) -> TModel:
        started = time.perf_counter()
        runnable = self._structured_runnable(schema)
        try:
            async with self._sem():
                result = await asyncio.wait_for(runnable.ainvoke(messages), self.settings.llm_timeout_seconds)
        except TimeoutError as exc:
            logger.warning("llm_timeout", extra={"task": task})
            raise LLMTimeoutError() from exc
        raw = result.get("raw") if isinstance(result, dict) else None
        _record(task, raw, started)
        try:
            parsed = result.get("parsed") if isinstance(result, dict) else result
            return _parse_structured(schema, raw, parsed)
        except (ValidationError, json.JSONDecodeError, LLMOutputError) as exc:
            logger.warning("llm_malformed_output", extra={"task": task, "error": type(exc).__name__})
            raise LLMOutputError() from exc

    async def atext(self, messages: list[BaseMessage], *, task: str) -> str:
        started = time.perf_counter()
        try:
            async with self._sem():
                message = await asyncio.wait_for(self.model.ainvoke(messages), self.settings.llm_timeout_seconds)
        except TimeoutError as exc:
            raise LLMTimeoutError() from exc
        _record(task, message, started)
        return str(message.content)

    async def astream_text(self, messages: list[BaseMessage], *, task: str) -> AsyncIterator[str]:
        started = time.perf_counter()
        final: Any = None
        async with self._sem():
            stream = self.model.astream(messages)
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(stream), self.settings.llm_timeout_seconds)
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    raise LLMTimeoutError() from exc
                final = chunk if final is None else final + chunk
                if chunk.content:
                    yield str(chunk.content)
        _record(task, final, started)

    # ------------------------------------------------------------------ sync
    def structured(self, schema: type[TModel], messages: list[BaseMessage], *, task: str) -> TModel:
        started = time.perf_counter()
        runnable = self._structured_runnable(schema)
        try:
            result = runnable.invoke(messages)
        except TimeoutError as exc:
            raise LLMTimeoutError() from exc
        raw = result.get("raw") if isinstance(result, dict) else None
        _record(task, raw, started)
        try:
            parsed = result.get("parsed") if isinstance(result, dict) else result
            return _parse_structured(schema, raw, parsed)
        except (ValidationError, json.JSONDecodeError, LLMOutputError) as exc:
            logger.warning("llm_malformed_output", extra={"task": task, "error": type(exc).__name__})
            raise LLMOutputError() from exc


def build_llm_client(settings: Settings) -> LLMClient | None:
    """Return an LLM client, or ``None`` when running with the heuristic provider."""
    if settings.resolved_llm_provider != "openai":
        return None
    return LLMClient(settings)
