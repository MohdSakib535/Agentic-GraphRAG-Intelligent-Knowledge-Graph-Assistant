"""Evaluation harness comparing Vector RAG, GraphRAG and Agentic GraphRAG.

Metrics per question:

* correctness       - fraction of expected answer keywords present in the answer
                      (unanswerable: 1.0 iff the insufficient-evidence answer was returned)
* faithfulness      - grounding score of the answer against its own evidence
* context_relevance - fraction of retrieved items containing at least one expected keyword
* retrieval_recall  - fraction of expected evidence keywords found in the retrieved context
* latency_ms, token_usage
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import select

from app.agents.nodes.common import Evidence, build_evidence, heuristic_answer, heuristic_verify, verbalize
from app.agents.state import INSUFFICIENT_EVIDENCE_ANSWER
from app.core.config import Settings, get_settings
from app.core.container import Container, build_container
from app.core.logging import get_logger
from app.db.postgres import utcnow
from app.llm.client import start_usage_tracking
from app.models.evaluation import EvaluationResult, EvaluationRun
from app.retrieval.query_parsing import expected_answer_type, relation_hints
from app.retrieval.types import RetrievalResult

logger = get_logger(__name__)

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "questions.json"
SYSTEMS = ("vector_rag", "graph_rag", "agentic_graphrag")


def load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    candidates = [path] if path else [DATASET_PATH, Path("/app/data/evaluation/questions.json"),
                                      Path(__file__).resolve().parents[3] / "data" / "evaluation" / "questions.json"]
    for candidate in candidates:
        if candidate and candidate.exists():
            return json.loads(candidate.read_text())["questions"]
    raise FileNotFoundError("Evaluation dataset not found")


def keyword_hit_rate(keywords: list[str], text: str) -> float:
    if not keywords:
        return 1.0
    lowered = text.lower()
    return sum(1 for k in keywords if k.lower() in lowered) / len(keywords)


def score_answer(item: dict[str, Any], answer: str, context_items: list[str], faithfulness: float) -> dict[str, float]:
    unanswerable = item["category"] == "unanswerable"
    insufficient = answer.strip() == INSUFFICIENT_EVIDENCE_ANSWER
    if unanswerable:
        correctness = 1.0 if insufficient else 0.0
        faithfulness = 1.0 if insufficient else faithfulness
        relevance = 1.0
        recall = 1.0
    else:
        correctness = 0.0 if insufficient else keyword_hit_rate(item.get("expected_keywords", []), answer)
        evidence_kw = item.get("evidence_keywords") or item.get("expected_keywords", [])
        joined = "\n".join(context_items)
        recall = keyword_hit_rate(evidence_kw, joined)
        relevance = (
            sum(1 for c in context_items if any(k.lower() in c.lower() for k in evidence_kw)) / len(context_items)
            if context_items else 0.0
        )
        if insufficient:
            faithfulness = 1.0  # abstaining is faithful, though not correct
    return {"correctness": round(correctness, 3), "faithfulness": round(faithfulness, 3),
            "context_relevance": round(relevance, 3), "retrieval_recall": round(recall, 3)}


def _context_items(result: RetrievalResult) -> list[str]:
    items = [c.text for c in result.chunks]
    items.extend(verbalize(f.model_dump()) + ". " + (f.evidence or "") for f in result.facts[:20])
    return items


async def _baseline(container: Container, system: str, item: dict[str, Any], tenant_id: str) -> dict[str, Any]:
    """Single-shot retrieval + grounded generation (no analysis loop, no grading, no rewriting)."""
    question = item["question"]
    strategy = "VECTOR" if system == "vector_rag" else "GRAPH"
    result = await container.retrieval.retrieve(
        strategy, question, tenant_id, relations=relation_hints(question), answer_type=expected_answer_type(question),
        use_cache=False,
    )
    state = {
        "question": question, "standalone_question": question, "tenant_id": tenant_id, "retrieval_strategy": strategy,
        "vector_results": [c.model_dump() for c in result.chunks], "graph_results": [f.model_dump() for f in result.facts],
        "linked_entities": [e.model_dump() for e in result.linked_entities],
        "answer_candidates": [c.name for c in result.answer_candidates], "relations": relation_hints(question),
        "cypher_rows": result.cypher_rows,
    }
    evidence: Evidence = await build_evidence(state, container.reader)
    if container.llm is not None and evidence.sources:
        from app.agents.nodes.generate_answer import _SYSTEM, _user_prompt
        from app.llm.client import to_messages

        answer = await container.llm.atext(to_messages(_SYSTEM, _user_prompt(state, evidence, False, [])), task=system)
        if answer.strip().startswith("INSUFFICIENT_EVIDENCE"):
            answer = INSUFFICIENT_EVIDENCE_ANSWER
    else:
        answer = heuristic_answer(state, evidence) if evidence.sources else ""
    answer = answer or INSUFFICIENT_EVIDENCE_ANSWER
    verification = heuristic_verify(question, answer, evidence, container.settings.verification_threshold)
    return {"answer": answer, "strategy": strategy, "context": _context_items(result),
            "faithfulness": verification["support_score"] if answer != INSUFFICIENT_EVIDENCE_ANSWER else 1.0}


async def _agentic(container: Container, item: dict[str, Any], tenant_id: str) -> dict[str, Any]:
    from app.agents.workflow import RECURSION_LIMIT, initial_turn_state, thread_id

    conv = f"eval-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id(tenant_id, conv), "tenant_id": tenant_id},
              "recursion_limit": RECURSION_LIMIT}
    state: dict[str, Any] = {}
    context: list[str] = []
    async for chunk in container.agent.astream(initial_turn_state(item["question"], tenant_id, conv, conv),
                                               config=config, stream_mode="updates", durability="exit"):
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            state.update(update)
            if node.endswith("_search"):
                context = [c.get("text", "") for c in update.get("vector_results") or []]
                context += [verbalize(f) + ". " + (f.get("evidence") or "") for f in (update.get("graph_results") or [])[:20]]
    try:
        await container.checkpointer.adelete_thread(thread_id(tenant_id, conv))
    except Exception:
        logger.debug("eval_thread_cleanup_failed")
    verification = state.get("verification") or {}
    support = verification.get("support_score")
    attempted = state.get("attempted_strategies") or [state.get("retrieval_strategy")]
    return {"answer": state.get("answer") or INSUFFICIENT_EVIDENCE_ANSWER, "strategy": attempted[0],
            "context": context, "faithfulness": float(support) if support is not None else 1.0}


async def evaluate_question(container: Container, system: str, item: dict[str, Any], tenant_id: str) -> dict[str, Any]:
    usage = start_usage_tracking()
    started = time.perf_counter()
    try:
        out = await (_agentic(container, item, tenant_id) if system == "agentic_graphrag"
                     else _baseline(container, system, item, tenant_id))
    except Exception as exc:
        logger.warning("evaluation_question_failed", extra={"system": system, "qid": item["id"], "error": type(exc).__name__})
        out = {"answer": INSUFFICIENT_EVIDENCE_ANSWER, "strategy": None, "context": [], "faithfulness": 0.0,
               "error": type(exc).__name__}
    latency = int((time.perf_counter() - started) * 1000)
    scores = score_answer(item, out["answer"], out["context"], out["faithfulness"])
    return {
        "system": system, "question_id": item["id"], "category": item["category"], "question": item["question"],
        "answer": out["answer"], "expected_strategy": item.get("expected_strategy"), "selected_strategy": out["strategy"],
        "latency_ms": latency, "token_usage": usage.total_tokens, **scores,
        "details": {"error": out.get("error"), "context_items": len(out["context"])},
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for system in sorted({r["system"] for r in results}):
        rows = [r for r in results if r["system"] == system]
        routed = [r for r in rows if r.get("expected_strategy")]
        summary[system] = {
            "questions": len(rows),
            "accuracy": round(mean(r["correctness"] for r in rows), 3),
            "faithfulness": round(mean(r["faithfulness"] for r in rows), 3),
            "context_relevance": round(mean(r["context_relevance"] for r in rows), 3),
            "retrieval_recall": round(mean(r["retrieval_recall"] for r in rows), 3),
            "avg_latency_ms": int(mean(r["latency_ms"] for r in rows)),
            "avg_token_usage": int(mean(r["token_usage"] for r in rows)),
            "routing_accuracy": (
                round(mean(1.0 if r["selected_strategy"] == r["expected_strategy"] else 0.0 for r in routed), 3)
                if system == "agentic_graphrag" and routed else None
            ),
            "by_category": {
                cat: round(mean(r["correctness"] for r in rows if r["category"] == cat), 3)
                for cat in sorted({r["category"] for r in rows})
            },
        }
    return summary


async def run_evaluation(container: Container, tenant_id: str, systems: list[str], categories: list[str] | None = None,
                         limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset()
    if categories:
        dataset = [q for q in dataset if q["category"] in categories]
    if limit:
        dataset = dataset[:limit]
    results = []
    for item in dataset:
        for system in systems:
            results.append(await evaluate_question(container, system, item, tenant_id))
    return results, summarise(results)


async def execute_evaluation_run(run_id: uuid.UUID, settings: Settings | None = None) -> dict[str, Any]:
    """Celery entry point. Uses its own, locally scoped connections (never the API's globals)."""
    import redis.asyncio as aioredis
    from neo4j import AsyncGraphDatabase
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    settings = settings or get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        container = build_container(settings, driver, redis_client)
        async with session_factory() as db:
            run = await db.get(EvaluationRun, run_id)
            if run is None:
                return {"status": "missing"}
            run.status = "RUNNING"
            await db.commit()
            tenant_id, systems = str(run.tenant_id), list(run.systems or SYSTEMS)
            options = dict(run.summary or {}).get("options", {})
        try:
            results, summary = await run_evaluation(container, tenant_id, systems, options.get("categories"),
                                                    options.get("limit"))
        except Exception as exc:
            logger.exception("evaluation_run_failed")
            async with session_factory() as db:
                run = await db.get(EvaluationRun, run_id)
                if run is not None:
                    run.status, run.error_message, run.finished_at = "FAILED", type(exc).__name__, utcnow()
                    await db.commit()
            return {"status": "failed"}
        async with session_factory() as db:
            run = await db.get(EvaluationRun, run_id)
            assert run is not None
            db.add_all(EvaluationResult(run_id=run.id, tenant_id=run.tenant_id, **r) for r in results)
            run.status, run.summary, run.finished_at = "COMPLETED", {"options": options, **summary}, utcnow()
            run.question_count = len({r["question_id"] for r in results})
            await db.commit()
        return {"status": "completed", "results": len(results)}
    finally:
        await redis_client.aclose()
        await driver.close()
        await engine.dispose()


async def latest_results(db: Any, tenant_id: uuid.UUID, run_id: uuid.UUID | None = None) -> tuple[EvaluationRun | None, list[EvaluationResult]]:
    query = select(EvaluationRun).where(EvaluationRun.tenant_id == tenant_id)
    if run_id:
        query = query.where(EvaluationRun.id == run_id)
    run = (await db.execute(query.order_by(EvaluationRun.created_at.desc()).limit(1))).scalar_one_or_none()
    if run is None:
        return None, []
    rows = await db.execute(
        select(EvaluationResult).where(EvaluationResult.run_id == run.id, EvaluationResult.tenant_id == tenant_id)
        .order_by(EvaluationResult.question_id, EvaluationResult.system)
    )
    return run, list(rows.scalars())
