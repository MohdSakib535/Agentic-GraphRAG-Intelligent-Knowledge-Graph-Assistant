"""Composition root: builds the object graph once per process / event loop.

The FastAPI lifespan builds one container; Celery tasks that need async
components (evaluation) build a short-lived one inside their own event loop.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from neo4j import AsyncDriver

from app.agents.nodes.common import AgentDeps
from app.agents.router import QueryAnalyzer
from app.agents.tools import AgentTools, build_tools
from app.agents.workflow import build_workflow
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.redis import TenantCache
from app.graph.repository import GraphReader
from app.graph.text2cypher import Text2Cypher
from app.ingestion.embedding import Embedder, build_embedder
from app.llm.client import LLMClient, build_llm_client
from app.retrieval.graph import GraphRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import build_reranker
from app.retrieval.retriever import RetrievalService
from app.retrieval.vector import VectorRetriever

logger = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    reader: GraphReader
    embedder: Embedder
    llm: LLMClient | None
    retrieval: RetrievalService
    analyzer: QueryAnalyzer
    tools: AgentTools
    agent: CompiledStateGraph
    checkpointer: BaseCheckpointSaver
    cache: TenantCache | None
    extras: dict[str, Any] = field(default_factory=dict)


def build_container(
    settings: Settings,
    driver: AsyncDriver | None,
    redis_client: Any | None,
    checkpointer: BaseCheckpointSaver | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    reader: Any | None = None,
) -> Container:
    """``reader`` may be injected (any object implementing the GraphReader interface) for tests."""
    if reader is None:
        if driver is None:
            raise ValueError("Either a Neo4j driver or a reader is required")
        reader = GraphReader(driver, settings)
    embedder = embedder or build_embedder(settings)
    llm = llm if llm is not None else build_llm_client(settings)
    cache = TenantCache(redis_client, settings.cache_ttl_seconds) if redis_client is not None else None
    text2cypher = Text2Cypher(llm, reader) if (llm is not None and settings.enable_text2cypher) else None
    vector = VectorRetriever(reader, embedder, settings)
    graph = GraphRetriever(reader, settings, text2cypher)
    hybrid = HybridRetriever(vector, graph, reader, settings)
    retrieval = RetrievalService(settings, reader, vector, graph, hybrid, build_reranker(settings), cache)
    analyzer = QueryAnalyzer(graph, llm)
    tools = build_tools(retrieval, reader, settings.tool_timeout_seconds)
    checkpointer = checkpointer or InMemorySaver()
    deps = AgentDeps(settings=settings, analyzer=analyzer, tools=tools, reader=reader, llm=llm)
    agent = build_workflow(deps, checkpointer)
    logger.info(
        "container_ready",
        extra={"llm_provider": settings.resolved_llm_provider, "embedding_provider": settings.resolved_embedding_provider,
               "reranker": settings.reranker, "text2cypher": text2cypher is not None},
    )
    return Container(settings, reader, embedder, llm, retrieval, analyzer, tools, agent, checkpointer, cache)


async def open_postgres_checkpointer(settings: Settings, stack: AsyncExitStack) -> BaseCheckpointSaver:
    """Persistent LangGraph checkpointer backed by PostgreSQL (connection pool)."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=settings.checkpoint_dsn,
        max_size=settings.db_pool_size,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    await pool.open()
    stack.push_async_callback(pool.close)
    saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
    await saver.setup()
    return saver
