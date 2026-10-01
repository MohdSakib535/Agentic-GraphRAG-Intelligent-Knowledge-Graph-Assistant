"""graph_search node - delegates to the graph_search_tool (tenant-scoped, timed, logged)."""

from __future__ import annotations

from typing import Any

from app.agents.nodes._retrieval import make_retrieval_node
from app.agents.nodes.common import AgentDeps


def make_graph_search(deps: AgentDeps) -> Any:
    return make_retrieval_node(deps, "GRAPH")
