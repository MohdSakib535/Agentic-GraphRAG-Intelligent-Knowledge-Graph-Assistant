"""Document-level permissions inside a tenant: retrieval, graph facts, agent answers, caching."""

from __future__ import annotations

import pytest
from agent_helpers import ask

from app.agents.state import INSUFFICIENT_EVIDENCE_ANSWER
from app.core.access import UNRESTRICTED, AccessScope, can_access, normalize_groups, require_scope, set_scope, use_scope
from app.core.errors import AuthorizationError, ValidationFailed
from conftest import TENANT_A


def doc_id(graph, filename: str) -> str:
    return next(d["id"] for d in graph.documents.values() if d["filename"] == filename and d["tenant_id"] == TENANT_A)


def test_group_rules() -> None:
    assert normalize_groups(["HR", "hr", " finance "]) == ["finance", "hr"]
    with pytest.raises(ValidationFailed):
        normalize_groups(["bad group!"])
    assert can_access([], ["x"], False) and can_access(["hr"], ["hr"], False)
    assert not can_access(["hr"], ["eng"], False) and can_access(["hr"], [], True)


def test_scope_fingerprint_and_fail_closed() -> None:
    a, b = AccessScope(("00000000-0000-0000-0000-000000000001",)), AccessScope(())
    assert a.fingerprint != b.fingerprint == UNRESTRICTED.fingerprint == "all"
    assert a.denied_chunk_prefixes == ["chk_00000000000000000000000000000001_"]
    with use_scope(a):
        assert require_scope() is a


async def test_restricted_document_is_invisible_to_retrieval(container, sample_graph) -> None:
    team = doc_id(sample_graph, "team-directory.md")
    set_scope(AccessScope((team,)))
    hits = await container.retrieval.vector.similarity_search("Amit reports to Rahul", TENANT_A, top_k=20)
    assert hits and all(h.document_id != team for h in hits)
    result = await container.retrieval.graph.search("Who reports to Rahul?", TENANT_A)
    assert not any(f.relationship == "REPORTS_TO" for f in result.facts)  # only stated in the restricted doc
    managed = await container.retrieval.graph.search("Who manages Project Alpha?", TENANT_A)
    fact = next(f for f in managed.facts if f.source == "Rahul" and f.target == "Project Alpha")
    assert team not in fact.document_ids  # visible via other documents, with restricted provenance removed
    set_scope(UNRESTRICTED)
    full = await container.retrieval.graph.search("Who reports to Rahul?", TENANT_A)
    assert {c.name for c in full.answer_candidates} == {"Amit", "Neha", "Priya"}


async def test_agent_answers_respect_document_permissions(container, sample_graph) -> None:
    team = doc_id(sample_graph, "team-directory.md")
    allowed = await ask(container, "Who reports to Rahul?")
    assert "Amit" in allowed["answer"]
    denied = await ask(container, "Who reports to Rahul?", denied=(team,))
    assert denied["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    still = await ask(container, "Who manages Project Alpha?", denied=(team,))
    assert still["answer"].startswith("Rahul")
    assert all(s.get("document_id") != team for s in still["sources"])


async def test_missing_scope_in_agent_config_fails_closed(container) -> None:
    from app.agents.workflow import RECURSION_LIMIT, initial_turn_state, thread_id

    config = {"configurable": {"thread_id": thread_id(TENANT_A, "x"), "tenant_id": TENANT_A},
              "recursion_limit": RECURSION_LIMIT}
    with pytest.raises(AuthorizationError):
        await container.agent.ainvoke(initial_turn_state("What is Kafka?", TENANT_A, "x", "t"), config=config)
