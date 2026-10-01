"""LangGraph workflow.

    START -> analyze_query -> route_retrieval -> {vector|graph|hybrid}_search -> grade_context
    grade_context --sufficient--> generate_answer
    grade_context --insufficient (retries < max)--> rewrite_query -> route_retrieval
    grade_context --insufficient (retries exhausted)--> generate_answer
    generate_answer -> verify_answer --fail (once)--> generate_answer
    verify_answer -> finalize -> END

Loops are bounded twice: by ``agent_max_retries`` / ``agent_max_regenerations``
in the routing functions, and by LangGraph's ``recursion_limit``.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.analyze_query import make_analyze_query
from app.agents.nodes.common import AgentDeps, step
from app.agents.nodes.generate_answer import make_generate_answer
from app.agents.nodes.grade_context import make_grade_context
from app.agents.nodes.graph_search import make_graph_search
from app.agents.nodes.hybrid_search import make_hybrid_search
from app.agents.nodes.rewrite_query import make_rewrite_query
from app.agents.nodes.vector_search import make_vector_search
from app.agents.nodes.verify_answer import make_verify_answer
from app.agents.state import AgentState

RETRIEVAL_NODES = {"VECTOR": "vector_search", "GRAPH": "graph_search", "HYBRID": "hybrid_search"}
# analyze + (retrieve + grade + rewrite) * (retries + 1) + (generate + verify) * 2 + finalize, with headroom.
RECURSION_LIMIT = 40


def route_retrieval(state: AgentState) -> str:
    strategy = state.get("retrieval_strategy", "HYBRID")
    if strategy == "DIRECT":
        return "generate_answer"
    return RETRIEVAL_NODES.get(strategy, "hybrid_search")


def make_after_grade(max_retries: int) -> Any:
    def after_grade(state: AgentState) -> str:
        if state.get("context_sufficient"):
            return "generate_answer"
        if state.get("retry_count", 0) < max_retries:
            return "rewrite_query"
        return "generate_answer"

    return after_grade


def after_verify(state: AgentState) -> str:
    verification = state.get("verification") or {}
    if verification.get("action") == "regenerate":
        return "generate_answer"
    return "finalize"


def make_finalize(deps: AgentDeps) -> Any:
    history_turns = deps.settings.conversation_history_turns

    async def finalize(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        history = list(state.get("history") or [])
        history.append({"role": "user", "content": state["question"][:1000]})
        history.append({"role": "assistant", "content": (state.get("answer") or "")[:600]})
        # Conversation memory: recent turns + entities in focus (for "that project", "he"...).
        focus: list[dict[str, str]] = []
        for e in state.get("linked_entities") or []:
            focus.append({"name": e["name"], "type": e["type"]})
        for fact in state.get("graph_results") or []:
            for name, etype in ((fact["source"], fact["source_type"]), (fact["target"], fact["target_type"])):
                if name in (state.get("answer_candidates") or []):
                    focus.append({"name": name, "type": etype})
        if not focus:
            focus = list(state.get("focus_entities") or [])
        dedup = list({(f["name"], f["type"]): f for f in focus}.values())[:10]
        return {
            "history": history[-2 * history_turns :],
            "focus_entities": dedup,
            # Bulky retrieval payloads are not persisted in the checkpoint (data minimisation).
            "retrieved_context": [],
            "vector_results": [],
            "graph_results": [],
            "evidence": {},
            "cypher_rows": [],
            "trace": step(state, "finalize", started),
        }

    return finalize


def build_workflow(deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("analyze_query", make_analyze_query(deps))
    graph.add_node("vector_search", make_vector_search(deps))
    graph.add_node("graph_search", make_graph_search(deps))
    graph.add_node("hybrid_search", make_hybrid_search(deps))
    graph.add_node("grade_context", make_grade_context(deps))
    graph.add_node("rewrite_query", make_rewrite_query(deps))
    graph.add_node("generate_answer", make_generate_answer(deps))
    graph.add_node("verify_answer", make_verify_answer(deps))
    graph.add_node("finalize", make_finalize(deps))

    graph.add_edge(START, "analyze_query")
    retrieval_targets = ["vector_search", "graph_search", "hybrid_search", "generate_answer"]
    graph.add_conditional_edges("analyze_query", route_retrieval, retrieval_targets)
    for node in RETRIEVAL_NODES.values():
        graph.add_edge(node, "grade_context")
    graph.add_conditional_edges(
        "grade_context", make_after_grade(deps.settings.agent_max_retries), ["generate_answer", "rewrite_query"]
    )
    graph.add_conditional_edges("rewrite_query", route_retrieval, retrieval_targets)
    graph.add_edge("generate_answer", "verify_answer")
    graph.add_conditional_edges("verify_answer", after_verify, ["generate_answer", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def thread_id(tenant_id: str, conversation_id: str) -> str:
    """Checkpoint thread ids embed the tenant so threads can never be shared across tenants."""
    return f"{tenant_id}:{conversation_id}"


def initial_turn_state(question: str, tenant_id: str, conversation_id: str, request_id: str) -> dict[str, Any]:
    """Per-turn input. Memory fields (history, focus_entities) are restored by the checkpointer."""
    return {
        "question": question,
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "request_id": request_id,
        "trace": [],
        "tools_called": [],
        "retrieval_latency_ms": 0,
        "retry_count": 0,
        "regeneration_count": 0,
        "verification": {},
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "context_sufficient": False,
        "cypher": None,
        "answer_candidates": [],
        "bridges": [],
    }
