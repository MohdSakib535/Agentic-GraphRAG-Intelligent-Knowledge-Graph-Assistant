"""Application error hierarchy and the consistent error-response contract.

Every error leaving the API has the shape::

    {"success": false, "error": {"code": "...", "message": "..."}, "request_id": "..."}

Stack traces are never returned to clients.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    code: str = "INTERNAL_ERROR"
    status_code: int = 500
    message: str = "An unexpected error occurred"

    def __init__(self, message: str | None = None, *, code: str | None = None, details: Any = None) -> None:
        self.message = message or self.message
        if code:
            self.code = code
        self.details = details
        super().__init__(self.message)


class ValidationFailed(AppError):
    code, status_code, message = "VALIDATION_ERROR", 422, "Request validation failed"


class AuthenticationError(AppError):
    code, status_code, message = "AUTHENTICATION_FAILED", 401, "Authentication failed"


class AuthorizationError(AppError):
    code, status_code, message = "FORBIDDEN", 403, "You do not have permission to perform this action"


class NotFoundError(AppError):
    code, status_code, message = "NOT_FOUND", 404, "Resource not found"


class ConflictError(AppError):
    code, status_code, message = "CONFLICT", 409, "Resource already exists"


class RateLimitExceeded(AppError):
    code, status_code, message = "RATE_LIMIT_EXCEEDED", 429, "Rate limit exceeded"

    def __init__(self, retry_after: int, message: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class InvalidFileError(AppError):
    code, status_code, message = "INVALID_FILE", 400, "The uploaded file is invalid"


class UnsupportedFileError(AppError):
    code, status_code, message = "UNSUPPORTED_FILE_TYPE", 415, "Unsupported file type"


class FileTooLargeError(AppError):
    code, status_code, message = "FILE_TOO_LARGE", 413, "The uploaded file is too large"


class DocumentProcessingError(AppError):
    code, status_code, message = "DOCUMENT_PROCESSING_FAILED", 500, "Document processing failed"


class LLMTimeoutError(AppError):
    code, status_code, message = "LLM_TIMEOUT", 504, "The language model did not respond in time"


class LLMOutputError(AppError):
    code, status_code, message = "MALFORMED_LLM_OUTPUT", 502, "The language model returned malformed output"


class EmbeddingError(AppError):
    code, status_code, message = "EMBEDDING_FAILED", 502, "Embedding generation failed"


class CypherValidationError(AppError):
    code, status_code, message = "CYPHER_VALIDATION_FAILED", 400, "Generated Cypher query was rejected"


class RetrievalError(AppError):
    code, status_code, message = "RETRIEVAL_FAILED", 502, "Retrieval failed"


class ServiceUnavailable(AppError):
    code, status_code, message = "SERVICE_UNAVAILABLE", 503, "A required service is unavailable"


class Neo4jUnavailable(ServiceUnavailable):
    code, message = "NEO4J_UNAVAILABLE", "The graph database is unavailable"


class PostgresUnavailable(ServiceUnavailable):
    code, message = "POSTGRES_UNAVAILABLE", "The relational database is unavailable"


class RedisUnavailable(ServiceUnavailable):
    code, message = "REDIS_UNAVAILABLE", "The cache/queue service is unavailable"


def error_payload(code: str, message: str, request_id: str | None, details: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return {"success": False, "error": error, "request_id": request_id}
