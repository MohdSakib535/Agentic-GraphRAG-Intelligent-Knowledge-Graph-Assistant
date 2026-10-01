"""Agent tools. Each tool validates input, enforces tenant isolation via the runnable
config, applies a timeout, returns structured data and logs its execution."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.agents.tools.document_tool import make_document_search_tool
from app.agents.tools.graph_tool import make_graph_search_tool
from app.agents.tools.hybrid_tool import make_hybrid_search_tool
from app.agents.tools.vector_tool import make_vector_search_tool
from app.graph.repository import GraphReader
from app.retrieval.retriever import RetrievalService


@dataclass
class AgentTools:
    vector_search: BaseTool
    graph_search: BaseTool
    hybrid_search: BaseTool
    document_search: BaseTool

    def for_strategy(self, strategy: str) -> BaseTool:
        return {"VECTOR": self.vector_search, "GRAPH": self.graph_search}.get(strategy, self.hybrid_search)


def build_tools(service: RetrievalService, reader: GraphReader, timeout: float) -> AgentTools:
    return AgentTools(
        vector_search=make_vector_search_tool(service, timeout),
        graph_search=make_graph_search_tool(service, timeout),
        hybrid_search=make_hybrid_search_tool(service, timeout),
        document_search=make_document_search_tool(service.vector, reader, timeout),
    )
