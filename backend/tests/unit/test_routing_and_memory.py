from __future__ import annotations

import pytest

from app.agents.router import candidate_entities, choose_strategy, resolve_coreferences
from app.retrieval.query_parsing import (
    expected_answer_type,
    information_terms,
    intermediate_type,
    primary_relation,
    relation_hints,
)
from app.retrieval.types import LinkedEntity


def linked(*names_types: tuple[str, str]) -> list[LinkedEntity]:
    return [LinkedEntity(id=n, name=n, type=t) for n, t in names_types]


@pytest.mark.parametrize(("question", "links", "expected"), [
    ("What is Kafka?", [("Kafka", "Technology")], "VECTOR"),
    ("Who manages Project Alpha?", [("Project Alpha", "Project")], "GRAPH"),
    ("Which developers work on Kafka projects managed by Rahul?", [("Kafka", "Technology"), ("Rahul", "Person")], "HYBRID"),
    ("Which technologies are shared between Project Alpha and Project Beta?",
     [("Project Alpha", "Project"), ("Project Beta", "Project")], "HYBRID"),
    ("How is Redis used in Project Alpha?", [("Redis", "Technology"), ("Project Alpha", "Project")], "HYBRID"),
    ("Tell me something not contained in the documents.", [], "VECTOR"),
    ("hello", [], "DIRECT"),
])
def test_strategy_selection(question: str, links: list[tuple[str, str]], expected: str) -> None:
    strategy, _, reasoning = choose_strategy(question, relation_hints(question), linked(*links), expected_answer_type(question))
    assert strategy == expected, reasoning


def test_routing_is_grounded_in_graph_linking() -> None:
    # Same wording, but the entity does not exist in this tenant's graph -> no GRAPH-only routing.
    strategy, _, _ = choose_strategy("Who manages Project Omega?", ["MANAGES"], [], "Person")
    assert strategy == "HYBRID"


def test_query_parsing_helpers() -> None:
    assert set(relation_hints("Who works on projects managed by Rahul that use Kafka?")) >= {"WORKS_ON", "MANAGES", "USES"}
    assert primary_relation("Who manages projects that use Kafka?") == "MANAGES"
    assert primary_relation("Who is Neha's manager?") == "REPORTS_TO"
    assert expected_answer_type("Which technologies does Project Alpha use?") == "Technology"
    assert expected_answer_type("What does Project Gamma build?") is None
    assert intermediate_type("Which technologies are used by projects that Priya manages?", ["Priya"], "Technology") == "Project"
    assert information_terms("What is the budget of Project Beta?", ["Project Beta"]) == {"budget"}
    assert information_terms("Who is Neha's manager?", ["Neha"]) == set()
    names = candidate_entities("Which developers work on Kafka projects managed by Rahul?")
    assert "Kafka" in names and "Rahul" in names


def test_coreference_resolution_from_conversation_memory() -> None:
    focus = [{"name": "Project Alpha", "type": "Project"}, {"name": "Rahul", "type": "Person"}]
    assert resolve_coreferences("What technologies does that project use?", focus) == \
        "What technologies does Project Alpha use?"
    assert resolve_coreferences("Who does he report to?", focus) == "Who does Rahul report to?"
    assert resolve_coreferences("Who works on it?", focus) == "Who works on Project Alpha?"
    assert resolve_coreferences("What is Kafka?", focus) == "What is Kafka?"
    assert resolve_coreferences("What does that project use?", []) == "What does that project use?"
