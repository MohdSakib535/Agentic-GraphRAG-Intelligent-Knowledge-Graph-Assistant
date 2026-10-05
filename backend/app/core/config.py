"""Application configuration loaded from environment variables.

All secrets come from the environment (or a local, git-ignored ``.env`` file).
Nothing secret is ever exposed through the public settings endpoint.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_RATE_RE = re.compile(r"^\s*(\d+)\s*/\s*(second|minute|hour|day)\s*$", re.IGNORECASE)
_PERIOD_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def parse_rate(rate: str) -> tuple[int, int]:
    """Parse a rate string such as ``"30/minute"`` into ``(limit, window_seconds)``."""
    match = _RATE_RE.match(rate)
    if not match:
        raise ValueError(f"Invalid rate limit expression: {rate!r} (expected e.g. '30/minute')")
    return int(match.group(1)), _PERIOD_SECONDS[match.group(2).lower()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ------------------------------------------------------------------ app
    app_name: str = "Agentic GraphRAG"
    environment: Literal["development", "test", "production"] = "development"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = True
    # Comma-separated in the environment (NoDecode: not parsed as JSON).
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["http://localhost:8501"])

    # ------------------------------------------------------------------ LLM
    # "auto" selects "openai" when an API key is configured, otherwise the
    # deterministic offline "heuristic" provider (rule-based extraction,
    # routing, grading and extractive answers).
    llm_provider: Literal["auto", "openai", "heuristic"] = "auto"
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_max_concurrency: int = 4

    embedding_provider: Literal["auto", "openai", "hashing"] = "auto"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64

    # ------------------------------------------------------------- Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "graphrag"
    postgres_user: str = "graphrag"
    postgres_password: SecretStr = SecretStr("graphrag")
    database_url_override: str | None = Field(default=None, alias="DATABASE_URL")
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # ---------------------------------------------------------------- Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("neo4jpassword")
    neo4j_database: str = "neo4j"
    neo4j_max_pool_size: int = 50
    neo4j_query_timeout_seconds: float = 15.0

    # ---------------------------------------------------------------- Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    cache_ttl_seconds: int = 600
    answer_cache_enabled: bool = True
    answer_cache_ttl_seconds: int = 1800
    embedding_cache_ttl_seconds: int = 86400
    celery_task_always_eager: bool = False

    # ------------------------------------------------------------------ JWT
    jwt_secret_key: SecretStr = SecretStr("change-me-in-production-please-32+chars")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    jwt_issuer: str = "agentic-graphrag"

    # --------------------------------------------------------------- Upload
    max_upload_size_mb: int = 25
    upload_dir: str = "/data/uploads"

    # ------------------------------------------------------- Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_auth: str = "10/minute"
    rate_limit_chat: str = "30/minute"
    rate_limit_upload: str = "10/hour"
    rate_limit_default: str = "120/minute"
    rate_limit_fail_open: bool = True

    # ------------------------------------------------------------ Retrieval
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 8
    vector_oversample_factor: int = 10
    retrieval_threshold: float = 0.55
    graph_max_hops: int = 3
    graph_max_facts: int = 60
    reranker: Literal["none", "score", "cross_encoder"] = "score"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    entity_similarity_threshold: float = 0.90
    entity_llm_resolution_band: float = 0.80
    enable_text2cypher: bool = True

    # ---------------------------------------------------------------- Agent
    agent_max_retries: int = 3
    agent_max_regenerations: int = 1
    tool_timeout_seconds: float = 20.0
    verification_threshold: float = 0.6
    conversation_history_turns: int = 6

    # ---------------------------------------------------------- Evaluation
    # LLM-as-judge scoring (correctness/faithfulness) when an LLM is configured; keyword scoring otherwise.
    eval_llm_judge: bool = True

    # ----------------------------------------------------------------- OCR
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    ocr_min_page_chars: int = 25  # pages with less extractable text than this are OCR'd

    # ------------------------------------------------------- Chat with CSV
    dataset_max_rows: int = 2_000_000
    dataset_query_timeout_seconds: float = 15.0
    dataset_result_limit: int = 200

    # ---------------------------------------------------------- Connectors
    # Fernet key for encrypting connector credentials at rest. Derived from JWT_SECRET_KEY when unset.
    encryption_key: SecretStr | None = None
    google_drive_api_url: str = "https://www.googleapis.com/drive/v3"
    google_token_url: str = "https://oauth2.googleapis.com/token"
    connector_max_files_per_sync: int = 500
    connector_sync_interval_minutes: int = 60  # Celery beat periodic sync (0 disables)

    # --------------------------------------------------- Google login (OIDC)
    google_client_id: str | None = None
    google_client_secret: SecretStr | None = None
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/google/callback"
    google_auth_url: str = "https://accounts.google.com/o/oauth2/v2/auth"
    google_jwks_url: str = "https://www.googleapis.com/oauth2/v3/certs"
    google_allowed_domains: Annotated[list[str], NoDecode] = Field(default_factory=list)
    frontend_url: str = "http://localhost:8501"

    # ------------------------------------------------------- Observability
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://jaeger:4318"
    otel_service_name: str = "agentic-graphrag-api"
    metrics_enabled: bool = True

    # ------------------------------------------------------------ Validators
    @field_validator("cors_origins", "google_allowed_domains", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            if value.strip().startswith("["):
                import json

                return json.loads(value)
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("rate_limit_auth", "rate_limit_chat", "rate_limit_upload", "rate_limit_default")
    @classmethod
    def _validate_rate(cls, value: str) -> str:
        parse_rate(value)
        return value

    @model_validator(mode="after")
    def _validate_security(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.environment == "production":
            secret = self.jwt_secret_key.get_secret_value()
            if len(secret) < 32 or secret.startswith("change-me"):
                raise ValueError("JWT_SECRET_KEY must be a strong secret (>= 32 chars) in production")
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS origins are not allowed in production")
        return self

    # ------------------------------------------------------------- Computed
    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def checkpoint_dsn(self) -> str:
        """Plain libpq DSN used by the LangGraph Postgres checkpointer."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")

    @property
    def resolved_llm_provider(self) -> Literal["openai", "heuristic"]:
        if self.llm_provider == "auto":
            return "openai" if self.openai_api_key and self.openai_api_key.get_secret_value() else "heuristic"
        return self.llm_provider

    @property
    def resolved_embedding_provider(self) -> Literal["openai", "hashing"]:
        if self.embedding_provider == "auto":
            return "openai" if self.openai_api_key and self.openai_api_key.get_secret_value() else "hashing"
        return self.embedding_provider

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def google_login_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    def public_dict(self) -> dict[str, object]:
        """Non-secret configuration safe to expose to authenticated clients."""
        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "llm_provider": self.resolved_llm_provider,
            "llm_model": self.llm_model if self.resolved_llm_provider == "openai" else "heuristic-local",
            "embedding_provider": self.resolved_embedding_provider,
            "embedding_model": (
                self.embedding_model if self.resolved_embedding_provider == "openai" else "feature-hashing"
            ),
            "embedding_dimensions": self.embedding_dimensions,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_threshold": self.retrieval_threshold,
            "reranker": self.reranker,
            "graph_max_hops": self.graph_max_hops,
            "agent_max_retries": self.agent_max_retries,
            "text2cypher_enabled": self.enable_text2cypher,
            "max_upload_size_mb": self.max_upload_size_mb,
            "ocr_enabled": self.ocr_enabled,
            "answer_cache_enabled": self.answer_cache_enabled,
            "google_login_enabled": self.google_login_enabled,
            "observability": {"tracing": self.otel_enabled, "metrics": self.metrics_enabled},
            "rate_limits": {
                "auth": self.rate_limit_auth,
                "chat": self.rate_limit_chat,
                "upload": self.rate_limit_upload,
            },
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
