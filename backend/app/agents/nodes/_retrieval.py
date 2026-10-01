"""Shared implementation of the strategy-specific retrieval nodes."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.nodes.common import AgentDeps, emit, step


def make_retrieval_node(deps: AgentDeps, strategy: str) -> Any:
    tool = deps.tools.for_strategy(strategy)

    async def retrieve(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        query = state.get("rewritten_query") or state.get("standalone_question") or state["question"]
        relaxed = strategy == "HYBRID" and state.get("retry_count", 0) >= 2
        emit("retrieval_started", {"strategy": strategy, "query": query, "tool": tool.name, "attempt": state.get("retry_count", 0) + 1})
        args: dict[str, Any] = {
            "query": query,
            "entities": state.get("entities") or [],
            "relations": [] if relaxed else (state.get("relations") or []),
            "answer_type": state.get("answer_type"),
            "top_k": deps.settings.top_k,
        }
        if state.get("metadata_filters"):
            args["filters"] = state["metadata_filters"]
        output = await tool.ainvoke(args, config=config)
        data = output.get("data") or {}
        chunks = data.get("chunks") or []
        facts = data.get("facts") or []
        linked = data.get("linked_entities") or state.get("linked_entities") or []
        candidates = [c["name"] for c in data.get("answer_candidates") or []]
        bridges = [{"name": b["name"], "type": b["type"]} for b in data.get("bridges") or []]
        errors = list(data.get("errors") or [])
        if not output.get("ok"):
            errors.append((output.get("error") or {}).get("code", "TOOL_FAILED"))
        latency = int((time.perf_counter() - started) * 1000)
        emit("retrieval_completed", {
            "strategy": strategy, "chunks": len(chunks), "facts": len(facts), "latency_ms": latency,
            "linked_entities": [e["name"] for e in linked], "answer_candidates": candidates,
            "cypher": data.get("cypher"), "cached": data.get("cached", False), "errors": errors,
        })
        return {
            "vector_results": chunks,
            "graph_results": facts,
            "retrieved_context": chunks,
            "linked_entities": linked,
            "answer_candidates": candidates,
            "bridges": bridges,
            "cypher": data.get("cypher"),
            "cypher_rows": data.get("cypher_rows") or [],
            "retrieval_errors": errors,
            "attempted_strategies": [*(state.get("attempted_strategies") or []), strategy],
            "tools_called": [*(state.get("tools_called") or []), tool.name],
            "retrieval_latency_ms": (state.get("retrieval_latency_ms") or 0) + latency,
            "trace": step(state, f"{strategy.lower()}_search", started, status="done" if output.get("ok") else "error",
                          chunks=len(chunks), facts=len(facts), candidates=candidates[:10], errors=errors,
                          cached=data.get("cached", False)),
        }

    retrieve.__name__ = f"{strategy.lower()}_search"
    return retrieve
