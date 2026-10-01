"""Evaluation harness tests: runs the 38-question benchmark against the in-memory sample corpus."""

from __future__ import annotations

from app.agents.state import INSUFFICIENT_EVIDENCE_ANSWER
from app.services.evaluation_service import load_dataset, run_evaluation, score_answer, summarise
from conftest import TENANT_A


def test_dataset_covers_required_categories() -> None:
    questions = load_dataset()
    assert len(questions) >= 30
    categories = {q["category"] for q in questions}
    assert categories == {"simple_factual", "semantic", "graph_relationship", "multi_hop", "hybrid", "unanswerable"}
    assert len({q["id"] for q in questions}) == len(questions)


def test_scoring_rules() -> None:
    answerable = {"category": "graph_relationship", "expected_keywords": ["Rahul"]}
    assert score_answer(answerable, "Rahul manages it [1]", ["Rahul manages Project Alpha"], 1.0)["correctness"] == 1.0
    assert score_answer(answerable, INSUFFICIENT_EVIDENCE_ANSWER, [], 1.0)["correctness"] == 0.0
    unanswerable = {"category": "unanswerable", "expected_keywords": []}
    assert score_answer(unanswerable, INSUFFICIENT_EVIDENCE_ANSWER, [], 0.0)["correctness"] == 1.0
    assert score_answer(unanswerable, "The CEO earns 1M", ["x"], 0.2)["correctness"] == 0.0


async def test_agentic_graphrag_outperforms_baselines(container) -> None:
    results, summary = await run_evaluation(container, TENANT_A, ["vector_rag", "graph_rag", "agentic_graphrag"])
    agentic, vector, graph = summary["agentic_graphrag"], summary["vector_rag"], summary["graph_rag"]
    assert agentic["questions"] == len(load_dataset())
    assert agentic["accuracy"] >= 0.9
    assert agentic["routing_accuracy"] >= 0.9
    assert agentic["by_category"]["unanswerable"] == 1.0  # never fabricates
    assert agentic["faithfulness"] >= 0.9
    assert agentic["accuracy"] > vector["accuracy"] and agentic["accuracy"] > graph["accuracy"]
    assert summarise(results) == summary
