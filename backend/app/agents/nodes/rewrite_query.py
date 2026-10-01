"""rewrite_query node: reformulate the query and escalate the retrieval strategy."""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.nodes.common import AgentDeps, emit, step
from app.core.errors import AppError
from app.llm.client import to_messages
from app.utils.text import content_terms

# Escalation ladder per attempt: broaden from precise to recall-oriented retrieval.
ESCALATION = {
    "VECTOR": ["HYBRID", "GRAPH", "HYBRID"],
    "GRAPH": ["HYBRID", "VECTOR", "HYBRID"],
    "HYBRID": ["HYBRID", "VECTOR", "GRAPH"],
}


class RewriteDecision(BaseModel):
    rewritten_query: str = Field(min_length=1, max_length=1000)
    retrieval_strategy: Literal["VECTOR", "GRAPH", "HYBRID"]
    reasoning: str = Field(default="", max_length=400)


def heuristic_rewrite(state: dict[str, Any]) -> tuple[str, str]:
    attempt = state.get("retry_count", 0)
    base = state.get("standalone_question") or state["question"]
    current = state.get("retrieval_strategy", "HYBRID")
    strategy = ESCALATION.get(current, ESCALATION["HYBRID"])[min(attempt, 2)]
    names = [e["name"] for e in state.get("linked_entities") or []]
    terms = list(dict.fromkeys(content_terms(base)))
    if attempt == 0:
        # Expand with linked graph entity names (canonical forms) and the answer type.
        extra = [n for n in names if n.lower() not in base.lower()]
        query = " ".join([base, *extra, state.get("answer_type") or ""]).strip()
    elif attempt == 1:
        # Keyword-focused reformulation.
        query = " ".join(dict.fromkeys(names + terms)) or base
    else:
        query = base
    return query, strategy


def make_rewrite_query(deps: AgentDeps) -> Any:
    async def rewrite_query(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        attempt = state.get("retry_count", 0)
        query, strategy = heuristic_rewrite(state)
        method = "heuristic"
        if deps.llm is not None:
            prompt = (
                f"Original question: {state.get('standalone_question')}\n"
                f"Previous query: {state.get('rewritten_query') or state.get('standalone_question')}\n"
                f"Previous strategies: {', '.join(state.get('attempted_strategies') or [])}\n"
                f"Missing information: {state.get('grade_reasoning') or 'unknown'}\n"
                f"Known graph entities: {', '.join(e['name'] for e in state.get('linked_entities') or []) or 'none'}\n"
                "Rewrite the query to retrieve better evidence and choose the next strategy."
            )
            try:
                decision = await deps.llm.astructured(
                    RewriteDecision,
                    to_messages("You improve search queries for a GraphRAG retrieval system. Keep the meaning; "
                                "add synonyms or canonical entity names; never add facts.", prompt),
                    task="rewrite_query",
                )
                query, strategy, method = decision.rewritten_query, decision.retrieval_strategy, "llm"
            except AppError:
                method = "heuristic(llm_failed)"
        emit("reasoning", {"message": f"Evidence insufficient - rewriting query (attempt {attempt + 1})",
                           "rewritten_query": query, "strategy": strategy})
        return {
            "rewritten_query": query,
            "retrieval_strategy": strategy,
            "retry_count": attempt + 1,
            "trace": step(state, "rewrite_query", started, rewritten_query=query, strategy=strategy,
                          attempt=attempt + 1, method=method),
        }

    return rewrite_query
