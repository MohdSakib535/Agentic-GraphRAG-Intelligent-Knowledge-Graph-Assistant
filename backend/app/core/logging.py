"""Structured logging with request-scoped context and secret redaction."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
tenant_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("tenant_id", default=None)
user_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)

_SENSITIVE_KEYS = re.compile(
    r"(password|passwd|secret|token|authorization|api[_-]?key|cookie|jwt|credential)", re.IGNORECASE
)
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9\-_\.=]+", re.IGNORECASE)
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{10,}")

# Attributes present on every LogRecord; anything else came from ``extra=``.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def _is_count_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in {"tokens", "token_usage", "token_count"} or lowered.endswith(("_tokens", "token_usage"))


def redact(value: Any, key: str | None = None) -> Any:
    """Recursively redact secrets from structures before they are logged."""
    if key is not None and _SENSITIVE_KEYS.search(key) and not _is_count_key(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = _BEARER_RE.sub(r"\1[REDACTED]", value)
        value = _JWT_RE.sub("[REDACTED_JWT]", value)
        return _OPENAI_KEY_RE.sub("[REDACTED_KEY]", value)
    return value


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        record.tenant_id = tenant_id_ctx.get()
        record.user_id = user_id_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
            "request_id": getattr(record, "request_id", None),
            "tenant_id": getattr(record, "tenant_id", None),
            "user_id": getattr(record, "user_id", None),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = redact(value, key)
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
        extras.pop("request_id", None)
        rid = getattr(record, "request_id", None)
        suffix = f" {json.dumps(redact(extras), default=str)}" if extras else ""
        return f"{redact(base)} rid={rid}{suffix}"


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Quiet noisy libraries; their DEBUG output may include payloads.
    for noisy in ("httpx", "httpcore", "openai", "botocore", "boto3", "neo4j", "urllib3", "multipart", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
