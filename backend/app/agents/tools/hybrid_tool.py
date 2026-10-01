from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from app.agents.tools._inputs import RetrievalToolInput, as_filter_dict
from app.agents.tools.base import run_tool
from app.retrieval.retriever import RetrievalService


def make_hybrid_search_tool(service: RetrievalService, timeout: float) -> BaseTool:
    @tool("hybrid_search_tool", args_schema=RetrievalToolInput)
    async def hybrid_search_tool(
        query: str, config: RunnableConfig, entities: list[str] | None = None, relations: list[str] | None = None,
        answer_type: str | None = None, top_k: int = 8, filters: Any = None,
    ) -> dict[str, Any]:
        """Hybrid retrieval: vector + keyword + graph traversal + metadata filters, fused with RRF."""

        async def call(tenant_id: str) -> dict[str, Any]:
            result = await service.retrieve(
                "HYBRID", query, tenant_id, entities=entities, relations=relations, answer_type=answer_type,
                top_k=top_k, filters=as_filter_dict(filters),
            )
            return result.model_dump(mode="json")

        return await run_tool("hybrid_search_tool", config, timeout, call)

    return hybrid_search_tool
