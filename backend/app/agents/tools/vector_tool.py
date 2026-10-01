from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from app.agents.tools._inputs import RetrievalToolInput, as_filter_dict
from app.agents.tools.base import run_tool
from app.retrieval.retriever import RetrievalService


def make_vector_search_tool(service: RetrievalService, timeout: float) -> BaseTool:
    @tool("vector_search_tool", args_schema=RetrievalToolInput)
    async def vector_search_tool(
        query: str, config: RunnableConfig, entities: list[str] | None = None, relations: list[str] | None = None,
        answer_type: str | None = None, top_k: int = 8, filters: Any = None,
    ) -> dict[str, Any]:
        """Semantic similarity search over document chunks (definitions, explanations, descriptions)."""

        async def call(tenant_id: str) -> dict[str, Any]:
            result = await service.retrieve(
                "VECTOR", query, tenant_id, top_k=top_k, filters=as_filter_dict(filters)
            )
            return result.model_dump(mode="json")

        return await run_tool("vector_search_tool", config, timeout, call)

    return vector_search_tool
