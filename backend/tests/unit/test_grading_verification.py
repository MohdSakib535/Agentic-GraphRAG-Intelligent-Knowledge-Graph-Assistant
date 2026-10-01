from __future__ import annotations

from app.agents.nodes.common import Evidence, heuristic_verify
from app.agents.nodes.grade_context import heuristic_grade

KAFKA_CHUNK = {"chunk_id": "c1", "document_id": "d1", "score": 0.9, "source_filename": "glossary.txt",
               "text": "Kafka is a distributed event streaming platform used for real-time data pipelines."}


def test_grade_sufficient_for_supported_question() -> None:
    state = {"question": "What is Kafka?", "standalone_question": "What is Kafka?", "retrieval_strategy": "VECTOR",
             "vector_results": [KAFKA_CHUNK], "graph_results": [], "entities": ["Kafka"],
             "linked_entities": [{"name": "Kafka", "type": "Technology"}]}
    grade, detail = heuristic_grade(state)
    assert grade >= 0.55, detail


def test_grade_insufficient_when_information_missing() -> None:
    state = {"question": "What is the salary of Kafka's creator?", "standalone_question": "What is the salary of Kafka's creator?",
             "retrieval_strategy": "VECTOR", "vector_results": [KAFKA_CHUNK], "graph_results": [], "entities": ["Kafka"],
             "linked_entities": [{"name": "Kafka", "type": "Technology"}]}
    grade, detail = heuristic_grade(state)
    assert grade < 0.55 and detail["information_coverage"] == 0.0


def test_grade_insufficient_when_named_entity_absent() -> None:
    state = {"question": "Who manages Project Omega?", "standalone_question": "Who manages Project Omega?",
             "retrieval_strategy": "HYBRID", "entities": ["Project Omega"], "linked_entities": [],
             "vector_results": [{**KAFKA_CHUNK, "text": "Rahul manages Project Alpha."}], "graph_results": []}
    grade, _ = heuristic_grade(state)
    assert grade < 0.55


def test_graph_grade_uses_answer_candidates() -> None:
    fact = {"source": "Rahul", "source_type": "Person", "relationship": "MANAGES", "target": "Project Alpha",
            "target_type": "Project", "chunk_ids": ["c1"]}
    state = {"question": "Who manages Project Alpha?", "standalone_question": "Who manages Project Alpha?",
             "retrieval_strategy": "GRAPH", "graph_results": [fact], "vector_results": [], "relations": ["MANAGES"],
             "answer_candidates": ["Rahul"], "linked_entities": [{"name": "Project Alpha", "type": "Project"}]}
    grade, _ = heuristic_grade(state)
    assert grade >= 0.9


def _evidence() -> Evidence:
    src = {"index": 1, "kind": "chunk", "chunk_id": "c1", "full_text": KAFKA_CHUNK["text"], "snippet": ""}
    return Evidence(sources=[src], chunks=[KAFKA_CHUNK], facts=[], prompt_text="")


def test_verification_passes_grounded_answer() -> None:
    result = heuristic_verify("What is Kafka?", "Kafka is a distributed event streaming platform. [1]", _evidence(), 0.6)
    assert result["passed"] and result["support_score"] == 1.0


def test_verification_fails_unsupported_or_uncited_answers() -> None:
    hallucinated = heuristic_verify("What is Kafka?",
                                    "Kafka was invented by Jay Kreps at LinkedIn in 2011 using Scala. [1]", _evidence(), 0.6)
    assert not hallucinated["passed"] and hallucinated["unsupported_claims"]
    uncited = heuristic_verify("What is Kafka?", "Kafka is a distributed event streaming platform.", _evidence(), 0.6)
    assert not uncited["passed"] and not uncited["has_citations"]
    bad_citation = heuristic_verify("What is Kafka?", "Kafka is a distributed event streaming platform. [7]", _evidence(), 0.6)
    assert not bad_citation["passed"]
