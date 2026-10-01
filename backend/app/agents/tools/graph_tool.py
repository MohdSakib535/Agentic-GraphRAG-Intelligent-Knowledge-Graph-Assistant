from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from app.agents.tools._inputs import RetrievalToolInput
from app.agents.tools.base import run_tool
from app.retrieval.retriever import RetrievalService


def make_graph_search_tool(service: RetrievalService, timeout: float) -> BaseTool:
    @tool("graph_search_tool", args_schema=RetrievalToolInput)
    async def graph_search_tool(
        query: str, config: RunnableConfig, entities: list[str] | None = None, relations: list[str] | None = None,
        answer_type: str | None = None, top_k: int = 8, filters: Any = None,
    ) -> dict[str, Any]:
        """Knowledge-graph lookup: links entities and traverses relationships (multi-hop, bounded)."""

        async def call(tenant_id: str) -> dict[str, Any]:
            result = await service.retrieve(
                "GRAPH", query, tenant_id, entities=entities, relations=relations, answer_type=answer_type, top_k=top_k
            )
            return result.model_dump(mode="json")

        return await run_tool("graph_search_tool", config, timeout, call)

    return graph_search_tool
