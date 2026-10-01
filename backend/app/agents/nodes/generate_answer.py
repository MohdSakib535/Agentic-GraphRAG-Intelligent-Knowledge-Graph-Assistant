"""generate_answer node: grounded answer generation with inline citations.

The model only ever sees the question plus retrieved evidence (graph facts,
passages and their source metadata). If the evidence is insufficient the node
returns the fixed insufficient-evidence answer instead of guessing.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.nodes.common import (
    AgentDeps,
    Evidence,
    build_evidence,
    cited_indices,
    emit,
    heuristic_answer,
    public_sources,
    step,
)
from app.agents.state import INSUFFICIENT_EVIDENCE_ANSWER
from app.core.errors import AppError
from app.llm.client import to_messages

DIRECT_ANSWER = (
    "Hi! I'm your knowledge-graph assistant. Ask me about the people, projects, technologies and documents "
    "in your uploaded knowledge base - for example \"Who manages Project Alpha?\" or \"What is Kafka?\"."
)
_INSUFFICIENT_SENTINEL = "INSUFFICIENT_EVIDENCE"

_SYSTEM = f"""You are an enterprise knowledge assistant. Answer ONLY from the numbered evidence provided.

Rules:
- Every factual sentence must end with citations like [1] or [2][3] referring to the evidence numbers.
- Never use outside knowledge, never guess, never invent names, numbers or relationships.
- Prefer knowledge-graph facts for relationship questions; use passages for explanations.
- Be concise: lead with the direct answer, then supporting facts.
- If the evidence does not contain the answer, reply with exactly: {_INSUFFICIENT_SENTINEL}"""


def _user_prompt(state: dict[str, Any], evidence: Evidence, strict: bool, unsupported: list[str]) -> str:
    question = state.get("standalone_question") or state["question"]
    extra = ""
    if strict:
        extra = ("\nYour previous answer contained claims not supported by the evidence"
                 + (f" ({'; '.join(unsupported[:3])})" if unsupported else "")
                 + ". Answer again using only explicitly stated facts.")
    hints = ""
    if state.get("answer_candidates"):
        hints = f"\nGraph traversal answer candidates: {', '.join(state['answer_candidates'][:10])}"
    return f"Evidence:\n{evidence.prompt_text}\n{hints}\n\nQuestion: {question}{extra}"


def make_generate_answer(deps: AgentDeps) -> Any:
    async def generate_answer(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        regen = state.get("regeneration_count", 0)
        emit("reasoning", {"message": "Generating answer..." if not regen else "Regenerating a stricter answer..."})
        if state.get("retrieval_strategy") == "DIRECT":
            answer, sources, evidence = DIRECT_ANSWER, [], None
            method = "direct"
        else:
            evidence = await build_evidence(state, deps.reader)
            if not state.get("context_sufficient") or not evidence.sources:
                answer, sources, method = INSUFFICIENT_EVIDENCE_ANSWER, [], "insufficient"
            elif deps.llm is not None:
                answer, method = await _llm_answer(deps, state, evidence, regen > 0), "llm"
            else:
                answer, method = _heuristic(state, evidence, strict=regen > 0), "heuristic"
                for token in answer.split(" "):
                    emit("token", {"text": token + " "})
            if answer.strip().startswith(_INSUFFICIENT_SENTINEL) or not answer.strip():
                answer, method = INSUFFICIENT_EVIDENCE_ANSWER, f"{method}:insufficient"
            if answer == INSUFFICIENT_EVIDENCE_ANSWER:
                sources = []
            else:
                cited = cited_indices(answer)
                sources = public_sources(evidence, cited or None)
                for src in sources:
                    emit("citation", src)
        return {
            "answer": answer,
            "sources": sources,
            "evidence": asdict(evidence) if evidence is not None else {},
            "trace": step(state, "generate_answer", started, method=method, citations=len(sources),
                          regeneration=regen),
        }

    return generate_answer


def _heuristic(state: dict[str, Any], evidence: Evidence, strict: bool) -> str:
    answer = heuristic_answer(state, evidence)
    if strict and answer:
        # Keep only lines that carry a citation (drop connective/summary text).
        answer = "\n".join(line for line in answer.split("\n") if "[" in line or not line.strip()).strip()
    return answer


async def _llm_answer(deps: AgentDeps, state: dict[str, Any], evidence: Evidence, strict: bool) -> str:
    unsupported = (state.get("verification") or {}).get("unsupported_claims") or []
    messages = to_messages(_SYSTEM, _user_prompt(state, evidence, strict, unsupported))
    parts: list[str] = []
    try:
        async for token in deps.llm.astream_text(messages, task="generate_answer"):  # type: ignore[union-attr]
            parts.append(token)
            emit("token", {"text": token})
    except AppError:
        return INSUFFICIENT_EVIDENCE_ANSWER
    return "".join(parts).strip()
