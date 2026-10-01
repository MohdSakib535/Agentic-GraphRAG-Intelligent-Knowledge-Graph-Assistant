"""LangGraph agent tests: routing, retrieval, grading, rewriting, generation, verification and memory."""

from __future__ import annotations

import uuid
from typing import Any

from app.agents.state import INSUFFICIENT_EVIDENCE_ANSWER
from app.agents.workflow import RECURSION_LIMIT, initial_turn_state, thread_id
from app.retrieval.types import RetrievalResult
from conftest import TENANT_A, TENANT_B


async def ask(container, question: str, tenant: str = TENANT_A, conversation: str | None = None) -> dict[str, Any]:
    conversation = conversation or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id(tenant, conversation), "tenant_id": tenant},
              "recursion_limit": RECURSION_LIMIT}
    state: dict[str, Any] = {"events": []}
    async for mode, chunk in container.agent.astream(initial_turn_state(question, tenant, conversation, "test"),
                                                     config=config, stream_mode=["updates", "custom"]):
        if mode == "custom":
            state["events"].append(chunk["event"])
            continue
        for node, update in chunk.items():
            if node.endswith("_search"):
                state["retrieved"] = update
            if node != "finalize":
                state.update(update)
    return state


async def test_vector_query(container) -> None:
    result = await ask(container, "What is Kafka?")
    assert result["retrieval_strategy"] == "VECTOR"
    assert "distributed event streaming" in result["answer"]
    assert result["sources"] and result["verification"]["passed"]
    assert result["confidence"] > 0.6


async def test_graph_query(container) -> None:
    result = await ask(container, "Who manages Project Alpha?")
    assert result["retrieval_strategy"] == "GRAPH"
    assert result["answer"].startswith("Rahul")
    assert any(s.get("source_filename") for s in result["sources"])


async def test_hybrid_multi_hop_query(container) -> None:
    result = await ask(container, "Which developers work on Kafka projects managed by Rahul?")
    assert result["attempted_strategies"][0] == "HYBRID"
    assert "Amit" in result["answer"] and "Neha" in result["answer"]
    assert "Priya" not in result["answer"].split("\n")[0]
    assert "Rahul manages Project Alpha" in result["answer"]  # supporting multi-hop chain is cited


async def test_shared_technologies_multi_hop(container) -> None:
    result = await ask(container, "Which technologies are shared between Project Alpha and Project Beta?")
    first = result["answer"].split("\n")[0]
    assert "Redis" in first and "PostgreSQL" in first and "Kafka" not in first


async def test_unanswerable_query_returns_insufficient_evidence(container) -> None:
    for question in ("Tell me something not contained in the documents.", "What is the budget of Project Beta?",
                     "Who manages Project Omega?"):
        result = await ask(container, question)
        assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWER, question
        assert result["sources"] == [] and result["confidence"] == 0.0


async def test_retries_are_bounded(container) -> None:
    result = await ask(container, "What is the salary of the CEO of TechCorp?")
    assert result["retry_count"] == container.settings.agent_max_retries == 3
    assert len(result["attempted_strategies"]) == 4  # initial attempt + 3 rewrites, then stop
    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWER


async def test_query_requiring_rewrite_recovers(container, monkeypatch) -> None:
    """The first retrieval returns nothing (e.g. a cold index); the agent rewrites and retries."""
    original = container.retrieval.retrieve
    calls: list[str] = []

    async def flaky(strategy: str, question: str, tenant_id: str, **kwargs: Any) -> RetrievalResult:
        calls.append(strategy)
        if len(calls) == 1:
            return RetrievalResult(strategy=strategy, query=question)
        return await original(strategy, question, tenant_id, **kwargs)

    monkeypatch.setattr(container.retrieval, "retrieve", flaky)
    result = await ask(container, "What is Kafka?")
    assert result["retry_count"] == 1 and result["rewritten_query"]
    assert calls[0] == "VECTOR" and calls[1] == "HYBRID"  # strategy escalation after insufficient evidence
    assert "distributed event streaming" in result["answer"]
    steps = [t["step"] for t in result["trace"]]
    assert steps[:4] == ["analyze_query", "vector_search", "grade_context", "rewrite_query"]


async def test_conversation_memory_resolves_references(container) -> None:
    conversation = uuid.uuid4().hex
    first = await ask(container, "Who manages Project Alpha?", conversation=conversation)
    assert first["answer"].startswith("Rahul")
    second = await ask(container, "What technologies does that project use?", conversation=conversation)
    assert second["standalone_question"] == "What technologies does Project Alpha use?"
    line = second["answer"].split("\n")[0]
    assert all(t in line for t in ("Kafka", "Redis", "PostgreSQL", "FastAPI"))
    snapshot = await container.agent.aget_state({"configurable": {"thread_id": thread_id(TENANT_A, conversation)}})
    assert len(snapshot.values["history"]) == 4
    assert snapshot.values["vector_results"] == []  # bulky evidence is not persisted in checkpoints


async def test_stream_events_and_trace(container) -> None:
    result = await ask(container, "Which projects use Kafka?")
    for event in ("query_analyzed", "retrieval_started", "retrieval_completed", "reasoning", "token", "citation",
                  "verification"):
        assert event in result["events"], event
    assert [t["step"] for t in result["trace"]][-2:] == ["generate_answer", "verify_answer"]


async def test_agent_respects_tenant_isolation(container) -> None:
    result = await ask(container, "Who manages Project Alpha?", tenant=TENANT_B)
    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    own = await ask(container, "Who manages Project Zeta?", tenant=TENANT_B)
    assert own["answer"].startswith("Zed")


async def test_direct_route_for_small_talk(container) -> None:
    from app.agents.nodes.generate_answer import DIRECT_ANSWER

    result = await ask(container, "hello")
    assert result["retrieval_strategy"] == "DIRECT" and result["answer"] == DIRECT_ANSWER
    assert "retrieved" not in result  # no retrieval tool was called


async def test_tools_reject_missing_tenant_context(container) -> None:
    output = await container.tools.vector_search.ainvoke({"query": "Kafka"}, config={"configurable": {}})
    assert output["ok"] is False and output["error"]["code"] == "VALIDATION_ERROR"
    output = await container.tools.graph_search.ainvoke({"query": "Kafka"}, config={"configurable": {"tenant_id": "x"}})
    assert output["ok"] is False


async def test_memory_tracks_bridge_entities_for_follow_ups(container) -> None:
    conversation = uuid.uuid4().hex
    await ask(container, "Which developers work on Kafka projects managed by Rahul?", conversation=conversation)
    follow = await ask(container, "What technologies does that project use?", conversation=conversation)
    assert follow["standalone_question"] == "What technologies does Project Alpha use?"
    assert "Kafka" in follow["answer"] and "Redis" in follow["answer"]
