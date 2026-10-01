from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.agents.tools.base import run_tool
from app.graph.repository import GraphReader
from app.retrieval.vector import VectorRetriever


class DocumentToolInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    filenames: list[str] = Field(default_factory=list, max_length=20, description="Restrict to these source files")
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    top_k: int = Field(default=5, ge=1, le=20)


def make_document_search_tool(vector: VectorRetriever, reader: GraphReader, timeout: float) -> BaseTool:
    @tool("document_search_tool", args_schema=DocumentToolInput)
    async def document_search_tool(
        query: str, config: RunnableConfig, filenames: list[str] | None = None, page_from: int | None = None,
        page_to: int | None = None, top_k: int = 5,
    ) -> dict[str, Any]:
        """Keyword search within specific documents/pages (metadata-filtered full-text search)."""

        async def call(tenant_id: str) -> dict[str, Any]:
            filters = {"filenames": filenames or None, "page_from": page_from, "page_to": page_to}
            hits = await vector.keyword_search(query, tenant_id, top_k, filters)
            if not hits:
                hits = await vector.similarity_search(query, tenant_id, top_k, filters)
            return {"chunks": [h.model_dump(mode="json") for h in hits]}

        return await run_tool("document_search_tool", config, timeout, call)

    return document_search_tool
