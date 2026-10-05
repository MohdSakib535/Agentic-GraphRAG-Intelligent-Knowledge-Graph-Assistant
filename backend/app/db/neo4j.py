"""Neo4j driver lifecycle (async driver for the API, sync driver for workers)."""

from __future__ import annotations

from neo4j import AsyncDriver, AsyncGraphDatabase, Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from app.core.config import Settings, get_settings
from app.core.errors import Neo4jUnavailable
from app.core.logging import get_logger
from app.graph.schema import VECTOR_INDEX_NAME, schema_statements

logger = get_logger(__name__)

_async_driver: AsyncDriver | None = None
_sync_driver: Driver | None = None

TRANSIENT_ERRORS = (ServiceUnavailable, SessionExpired)


def _driver_kwargs(settings: Settings) -> dict[str, object]:
    return {
        "auth": (settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        "max_connection_pool_size": settings.neo4j_max_pool_size,
        "connection_acquisition_timeout": 10.0,
        "connection_timeout": 5.0,
    }


def init_async_driver(settings: Settings | None = None) -> AsyncDriver:
    global _async_driver
    settings = settings or get_settings()
    _async_driver = AsyncGraphDatabase.driver(settings.neo4j_uri, **_driver_kwargs(settings))
    return _async_driver


def get_async_driver() -> AsyncDriver:
    if _async_driver is None:
        return init_async_driver()
    return _async_driver


async def close_async_driver() -> None:
    global _async_driver
    if _async_driver is not None:
        await _async_driver.close()
    _async_driver = None


def get_sync_driver(settings: Settings | None = None) -> Driver:
    global _sync_driver
    if _sync_driver is None:
        settings = settings or get_settings()
        _sync_driver = GraphDatabase.driver(settings.neo4j_uri, **_driver_kwargs(settings))
    return _sync_driver


def close_sync_driver() -> None:
    global _sync_driver
    if _sync_driver is not None:
        _sync_driver.close()
    _sync_driver = None


async def ping_neo4j(driver: AsyncDriver | None = None) -> bool:
    driver = driver or get_async_driver()
    try:
        await driver.verify_connectivity()
    except (Neo4jError, *TRANSIENT_ERRORS, OSError) as exc:
        raise Neo4jUnavailable() from exc
    return True


INDEX_DIMENSIONS = (
    "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name "
    "RETURN options.indexConfig['vector.dimensions'] AS dimensions"
)
COUNT_EMBEDDED = "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS n"


def _mismatch_message(current: int, wanted: int) -> str:
    return (f"The Neo4j vector index has {current} dimensions but the configured embedding model produces "
            f"{wanted}. Run `python -m app.scripts.reindex_embeddings` (make reindex) to re-embed existing chunks.")


async def ensure_schema(driver: AsyncDriver, settings: Settings) -> None:
    """Create constraints and indexes idempotently; adapt an empty vector index to a new embedding size."""
    db = settings.neo4j_database
    for statement in schema_statements(settings.embedding_dimensions):
        await driver.execute_query(statement, database_=db)
    records, _, _ = await driver.execute_query(INDEX_DIMENSIONS, {"name": VECTOR_INDEX_NAME}, database_=db)
    current = int(records[0]["dimensions"]) if records and records[0]["dimensions"] is not None else None
    if current is not None and current != settings.embedding_dimensions:
        embedded, _, _ = await driver.execute_query(COUNT_EMBEDDED, database_=db)
        if embedded[0]["n"] == 0:  # nothing indexed yet (e.g. switched provider before ingesting): recreate
            await driver.execute_query(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS", database_=db)
            for statement in schema_statements(settings.embedding_dimensions):
                await driver.execute_query(statement, database_=db)
            logger.info("vector_index_recreated", extra={"from": current, "to": settings.embedding_dimensions})
        else:
            logger.error("embedding_dimension_mismatch", extra={"index_dimensions": current,
                                                                 "embedding_dimensions": settings.embedding_dimensions,
                                                                 "hint": _mismatch_message(current, settings.embedding_dimensions)})
    logger.info("neo4j_schema_ready", extra={"embedding_dimensions": settings.embedding_dimensions})


def ensure_schema_sync(driver: Driver, settings: Settings) -> None:
    for statement in schema_statements(settings.embedding_dimensions):
        driver.execute_query(statement, database_=settings.neo4j_database)


def vector_index_dimensions_sync(driver: Driver, settings: Settings) -> int | None:
    records, _, _ = driver.execute_query(INDEX_DIMENSIONS, {"name": VECTOR_INDEX_NAME},
                                         database_=settings.neo4j_database)
    return int(records[0]["dimensions"]) if records and records[0]["dimensions"] is not None else None
