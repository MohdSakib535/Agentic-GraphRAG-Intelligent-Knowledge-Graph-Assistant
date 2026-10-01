"""PostgreSQL access via SQLAlchemy 2 (async for the API, sync for Celery workers)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, MetaData, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import Settings, get_settings

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in unit tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONType, list[Any]: JSONType}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class UUIDPrimaryKey:
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)


# --------------------------------------------------------------------------- engines
_async_engine: AsyncEngine | None = None
_async_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_sync_engine: Engine | None = None
_sync_sessionmaker: sessionmaker[Session] | None = None


def _engine_kwargs(url: str, settings: Settings) -> dict[str, Any]:
    if url.startswith("sqlite"):
        return {}
    return {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }


def init_async_engine(settings: Settings | None = None, url: str | None = None) -> AsyncEngine:
    global _async_engine, _async_sessionmaker
    settings = settings or get_settings()
    url = url or settings.database_url
    _async_engine = create_async_engine(url, **_engine_kwargs(url, settings))
    _async_sessionmaker = async_sessionmaker(_async_engine, expire_on_commit=False)
    return _async_engine


def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _async_sessionmaker is None:
        init_async_engine()
    assert _async_sessionmaker is not None
    return _async_sessionmaker


async def dispose_async_engine() -> None:
    global _async_engine, _async_sessionmaker
    if _async_engine is not None:
        await _async_engine.dispose()
    _async_engine = None
    _async_sessionmaker = None


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a transactional async session."""
    async with get_async_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def ping_postgres() -> bool:
    async with get_async_sessionmaker()() as session:
        await session.execute(text("SELECT 1"))
    return True


def get_sync_sessionmaker(settings: Settings | None = None) -> sessionmaker[Session]:
    global _sync_engine, _sync_sessionmaker
    if _sync_sessionmaker is None:
        settings = settings or get_settings()
        url = settings.database_url
        _sync_engine = create_engine(url, **_engine_kwargs(url, settings))
        _sync_sessionmaker = sessionmaker(_sync_engine, expire_on_commit=False)
    return _sync_sessionmaker


@contextmanager
def sync_session_scope() -> Iterator[Session]:
    """Transactional scope for Celery tasks."""
    session = get_sync_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
