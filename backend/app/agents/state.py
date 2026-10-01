"""LangGraph agent state."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Strategy = Literal["VECTOR", "GRAPH", "HYBRID", "DIRECT"]

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough information in the uploaded knowledge base to answer this reliably."
)


class AgentState(TypedDict, total=False):
    # ---- request
    question: str
    tenant_id: str
    conversation_id: str
    request_id: str
    # ---- conversation memory (persisted by the checkpointer across turns)
    history: list[dict[str, str]]
    focus_entities: list[dict[str, str]]
    # ---- analysis
    standalone_question: str
    intent: str
    entities: list[str]
    relations: list[str]
    answer_type: str | None
    temporal_constraints: list[str]
    metadata_filters: dict[str, Any]
    retrieval_strategy: Strategy
    analysis_reasoning: str
    # ---- retrieval (cleared before the final checkpoint - see finalize)
    rewritten_query: str
    retrieved_context: list[dict[str, Any]]
    vector_results: list[dict[str, Any]]
    graph_results: list[dict[str, Any]]
    linked_entities: list[dict[str, Any]]
    answer_candidates: list[str]
    bridges: list[dict[str, str]]
    cypher: str | None
    cypher_rows: list[dict[str, Any]]
    retrieval_errors: list[str]
    attempted_strategies: list[str]
    # ---- grading / generation / verification
    context_grade: float
    context_sufficient: bool
    grade_reasoning: str
    evidence: dict[str, Any]
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    verification: dict[str, Any]
    retry_count: int
    regeneration_count: int
    # ---- observability (reset at the start of every turn; nodes run sequentially and append)
    tools_called: list[str]
    trace: list[dict[str, Any]]
    retrieval_latency_ms: int
