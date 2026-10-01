"""grade_context node: is the retrieved evidence sufficient to answer?

A cheap deterministic grade is always computed. The LLM grader is consulted only
when the deterministic grade falls in an ambiguous band - clear passes and clear
failures never cost an LLM call.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.nodes.common import AgentDeps, emit, key_terms, relevant_facts, step, verbalize
from app.core.errors import AppError
from app.llm.client import to_messages
from app.retrieval.query_parsing import information_terms
from app.utils.text import term_set, truncate

AMBIGUOUS_BAND = (0.3, 0.75)


class ContextGrade(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    sufficient: bool
    missing_information: str = Field(default="", max_length=400)


def heuristic_grade(state: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    question = state.get("rewritten_query") or state.get("standalone_question") or state["question"]
    strategy = state.get("retrieval_strategy", "HYBRID")
    chunks = state.get("vector_results") or []
    facts = state.get("graph_results") or []
    q_terms = key_terms(state.get("standalone_question") or question)
    top_text = " ".join(c.get("text", "") for c in chunks[:3])
    fact_text = " ".join(verbalize(f) for f in facts[:20])
    evidence_terms = term_set(top_text + " " + fact_text)
    term_cov = len(q_terms & evidence_terms) / len(q_terms) if q_terms else 0.0
    names = [e["name"] for e in state.get("linked_entities") or []] or (state.get("entities") or [])
    lowered = (top_text + " " + fact_text).lower()
    entity_cov = (sum(1 for n in names if n.lower() in lowered) / len(names)) if names else None
    top_score = max((float(c.get("score") or 0) for c in chunks), default=0.0)
    rel_facts = relevant_facts(state, facts)
    if state.get("answer_candidates"):
        graph_signal = 1.0
    elif rel_facts:
        graph_signal = 0.85
    elif state.get("cypher_rows"):
        graph_signal = 0.7
    elif facts:
        graph_signal = 0.35
    else:
        graph_signal = 0.0
    ec = entity_cov if entity_cov is not None else term_cov
    vector_grade = 0.6 * term_cov + 0.25 * min(1.0, top_score) + 0.15 * ec
    graph_grade = 0.7 * graph_signal + 0.3 * (entity_cov or 0.0)
    if strategy == "VECTOR":
        grade = vector_grade
    elif strategy == "GRAPH":
        grade = graph_grade
    else:
        grade = max(vector_grade, graph_grade)
    # Hallucination guards: named entities and the information asked for must appear in the evidence.
    if entity_cov is not None:
        grade *= 0.3 + 0.7 * entity_cov
    all_names = list({*names, *(state.get("entities") or [])})
    info = information_terms(state.get("standalone_question") or question, all_names)
    all_evidence = term_set(" ".join(c.get("text", "") for c in chunks) + " " + fact_text)
    info_cov = len(info & all_evidence) / len(info) if info else None
    if info_cov is not None and info_cov < 0.34:
        grade = min(grade, 0.3)
    detail = {
        "term_coverage": round(term_cov, 3), "entity_coverage": None if entity_cov is None else round(entity_cov, 3),
        "top_score": round(top_score, 3), "graph_signal": graph_signal, "relevant_facts": len(rel_facts),
        "chunks": len(chunks), "facts": len(facts), "information_terms": sorted(info),
        "information_coverage": None if info_cov is None else round(info_cov, 3),
    }
    return round(grade, 3), detail


def make_grade_context(deps: AgentDeps) -> Any:
    threshold = deps.settings.retrieval_threshold

    async def grade_context(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        emit("reasoning", {"message": "Grading evidence..."})
        grade, detail = heuristic_grade(state)
        method = "heuristic"
        reasoning = ""
        if deps.llm is not None and AMBIGUOUS_BAND[0] <= grade < AMBIGUOUS_BAND[1]:
            evidence = "\n".join(f"- {truncate(c.get('text', ''), 600)}" for c in (state.get("vector_results") or [])[:5])
            facts = "\n".join(f"- {verbalize(f)}" for f in (state.get("graph_results") or [])[:15])
            prompt = (f"Question: {state.get('standalone_question')}\n\nPassages:\n{evidence or '(none)'}\n\n"
                      f"Graph facts:\n{facts or '(none)'}\n\nCan the question be fully answered from this evidence alone?")
            try:
                verdict = await deps.llm.astructured(
                    ContextGrade,
                    to_messages("You grade retrieved evidence for a RAG system. Be strict: only mark sufficient "
                                "if the evidence explicitly contains the answer.", prompt),
                    task="grade_context",
                )
                grade = round(0.4 * grade + 0.6 * verdict.relevance, 3)
                if verdict.sufficient:
                    grade = max(grade, threshold)
                elif grade >= threshold:
                    grade = threshold - 0.01
                reasoning = verdict.missing_information
                method = "llm"
            except AppError:
                method = "heuristic(llm_failed)"
        sufficient = grade >= threshold
        emit("reasoning", {"message": f"Evidence grade {grade:.2f} ({'sufficient' if sufficient else 'insufficient'})",
                           "grade": grade, "sufficient": sufficient})
        return {
            "context_grade": grade,
            "context_sufficient": sufficient,
            "grade_reasoning": reasoning,
            "trace": step(state, "grade_context", started, grade=grade, sufficient=sufficient, method=method, **detail),
        }

    return grade_context
