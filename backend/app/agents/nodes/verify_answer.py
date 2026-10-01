"""verify_answer node: check grounding, citations, relevance and confidence.

On failure the workflow regenerates once (stricter prompt); if verification still
fails, the answer is replaced with the insufficient-evidence response.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.nodes.common import AgentDeps, Evidence, emit, heuristic_verify, step
from app.agents.nodes.generate_answer import DIRECT_ANSWER
from app.agents.state import INSUFFICIENT_EVIDENCE_ANSWER
from app.core.errors import AppError
from app.llm.client import to_messages
from app.utils.text import truncate


class LLMVerification(BaseModel):
    all_claims_supported: bool
    unsupported_claims: list[str] = Field(default_factory=list, max_length=10)
    introduces_outside_information: bool
    answers_the_question: bool
    confidence: float = Field(ge=0.0, le=1.0)


def make_verify_answer(deps: AgentDeps) -> Any:
    settings = deps.settings

    async def verify_answer(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        emit("reasoning", {"message": "Verifying answer..."})
        answer = state.get("answer") or ""
        grade = float(state.get("context_grade") or 0.0)
        regen = state.get("regeneration_count", 0)
        if answer in (INSUFFICIENT_EVIDENCE_ANSWER, DIRECT_ANSWER):
            result = {"passed": True, "method": "skipped", "support_score": None,
                      "reason": "insufficient-evidence response" if answer != DIRECT_ANSWER else "direct response"}
            confidence = 0.0 if answer == INSUFFICIENT_EVIDENCE_ANSWER else 1.0
            emit("verification", result)
            return {"verification": result, "confidence": confidence,
                    "trace": step(state, "verify_answer", started, **result)}

        ev = state.get("evidence") or {}
        evidence = Evidence(sources=ev.get("sources", []), chunks=ev.get("chunks", []), facts=ev.get("facts", []),
                            prompt_text=ev.get("prompt_text", ""))
        question = state.get("standalone_question") or state["question"]
        result = heuristic_verify(question, answer, evidence, settings.verification_threshold)
        if deps.llm is not None:
            try:
                verdict = await deps.llm.astructured(
                    LLMVerification,
                    to_messages(
                        "You verify that an answer is fully grounded in the provided evidence. Be strict.",
                        f"Evidence:\n{truncate(evidence.prompt_text, 12000)}\n\nQuestion: {question}\n\nAnswer:\n{answer}",
                    ),
                    task="verify_answer",
                )
                llm_ok = (verdict.all_claims_supported and not verdict.introduces_outside_information
                          and verdict.answers_the_question)
                result.update({
                    "method": "llm+heuristic",
                    "llm_confidence": verdict.confidence,
                    "unsupported_claims": verdict.unsupported_claims or result["unsupported_claims"],
                    "relevant": verdict.answers_the_question,
                    "passed": llm_ok and result["has_citations"],
                    "support_score": round(min(1.0, (result["support_score"] + verdict.confidence) / 2), 3),
                })
            except AppError:
                result["method"] = "heuristic(llm_failed)"
        support = float(result.get("support_score") or 0.0)
        confidence = round(0.4 * grade + 0.6 * support, 3)
        result["confidence_ok"] = confidence >= settings.verification_threshold * 0.8
        result["passed"] = bool(result["passed"] and result["confidence_ok"])
        update: dict[str, Any] = {"verification": result, "confidence": confidence}
        if not result["passed"]:
            if regen < settings.agent_max_regenerations:
                update["regeneration_count"] = regen + 1
                result["action"] = "regenerate"
            else:
                update.update({"answer": INSUFFICIENT_EVIDENCE_ANSWER, "sources": [], "confidence": 0.0})
                result["action"] = "replaced_with_insufficient_evidence"
        emit("verification", result)
        update["trace"] = step(state, "verify_answer", started, status="done" if result["passed"] else "failed",
                               passed=result["passed"], support=support, confidence=update["confidence"],
                               action=result.get("action"), method=result.get("method"))
        return update

    return verify_answer
