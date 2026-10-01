"""analyze_query node: intent, entities, relationships, temporal constraints, strategy."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.nodes.common import AgentDeps, emit, step
from app.agents.router import detect_metadata_filters


def make_analyze_query(deps: AgentDeps) -> Any:
    async def analyze_query(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        emit("reasoning", {"message": "Analyzing query..."})
        analysis, linked = await deps.analyzer.analyze(
            state["question"], state["tenant_id"], state.get("history") or [], state.get("focus_entities") or []
        )
        filters = detect_metadata_filters(analysis.standalone_question)
        payload = {
            "standalone_question": analysis.standalone_question,
            "intent": analysis.intent,
            "entities": analysis.entities,
            "relations": analysis.relationships,
            "answer_type": analysis.answer_type,
            "temporal_constraints": analysis.temporal_constraints,
            "retrieval_strategy": analysis.retrieval_strategy,
            "reasoning": analysis.reasoning,
            "linked_entities": [e.model_dump() for e in linked],
        }
        emit("query_analyzed", payload)
        return {
            "standalone_question": analysis.standalone_question,
            "intent": analysis.intent,
            "entities": analysis.entities,
            "relations": analysis.relationships,
            "answer_type": analysis.answer_type,
            "temporal_constraints": analysis.temporal_constraints,
            "metadata_filters": filters,
            "retrieval_strategy": analysis.retrieval_strategy,
            "analysis_reasoning": analysis.reasoning,
            "linked_entities": [e.model_dump() for e in linked],
            "retry_count": 0,
            "regeneration_count": 0,
            "attempted_strategies": [],
            "rewritten_query": "",
            "trace": step(state, "analyze_query", started, strategy=analysis.retrieval_strategy,
                          intent=analysis.intent, entities=analysis.entities, relations=analysis.relationships),
        }

    return analyze_query
