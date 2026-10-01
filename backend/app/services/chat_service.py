"""Chat orchestration: conversations, the LangGraph agent run, persistence and observability."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.workflow import RECURSION_LIMIT, initial_turn_state, thread_id
from app.core.container import Container
from app.core.errors import NotFoundError
from app.core.logging import get_logger, request_id_ctx
from app.db.postgres import utcnow
from app.llm.client import start_usage_tracking
from app.models.conversation import Conversation
from app.models.message import Message
from app.utils.text import truncate

logger = get_logger(__name__)

# State keys captured from node updates while streaming (they are cleared before checkpointing).
_CAPTURE = ("vector_results", "graph_results", "linked_entities", "answer_candidates", "bridges", "cypher", "cypher_rows",
            "retrieval_errors")


class ChatService:
    def __init__(self, db: AsyncSession, container: Container) -> None:
        self.db = db
        self.container = container

    async def get_or_create_conversation(self, tenant_id: uuid.UUID, user_id: uuid.UUID,
                                         conversation_id: uuid.UUID | None, first_message: str) -> Conversation:
        if conversation_id is not None:
            conv = (
                await self.db.execute(
                    select(Conversation).where(Conversation.id == conversation_id, Conversation.tenant_id == tenant_id,
                                               Conversation.user_id == user_id)
                )
            ).scalar_one_or_none()
            if conv is None:
                raise NotFoundError("Conversation not found", code="CONVERSATION_NOT_FOUND")
            return conv
        conv = Conversation(tenant_id=tenant_id, user_id=user_id, title=truncate(first_message.strip(), 80))
        self.db.add(conv)
        await self.db.commit()
        return conv

    async def list_conversations(self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int = 50) -> list[Conversation]:
        rows = await self.db.execute(
            select(Conversation).where(Conversation.tenant_id == tenant_id, Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc()).limit(limit)
        )
        return list(rows.scalars())

    async def messages(self, tenant_id: uuid.UUID, user_id: uuid.UUID, conversation_id: uuid.UUID) -> list[Message]:
        await self.get_or_create_conversation(tenant_id, user_id, conversation_id, "")
        rows = await self.db.execute(
            select(Message).where(Message.conversation_id == conversation_id, Message.tenant_id == tenant_id)
            .order_by(Message.created_at.asc())
        )
        return list(rows.scalars())

    async def delete_conversation(self, tenant_id: uuid.UUID, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        conv = await self.get_or_create_conversation(tenant_id, user_id, conversation_id, "")
        await self.db.delete(conv)
        await self.db.commit()
        try:
            await self.container.checkpointer.adelete_thread(thread_id(str(tenant_id), str(conversation_id)))
        except Exception:
            logger.warning("checkpoint_delete_failed")

    # ----------------------------------------------------------------- run
    async def stream_turn(self, tenant_id: uuid.UUID, user_id: uuid.UUID, conversation: Conversation,
                          message: str) -> AsyncIterator[dict[str, Any]]:
        """Run the agent, yielding SSE-style events; the final event is ``completed``."""
        started = time.perf_counter()
        usage = start_usage_tracking()
        request_id = request_id_ctx.get() or uuid.uuid4().hex
        tid = str(tenant_id)
        user_msg = Message(tenant_id=tenant_id, conversation_id=conversation.id, role="user", content=message)
        self.db.add(user_msg)
        await self.db.commit()

        config = {
            "configurable": {"thread_id": thread_id(tid, str(conversation.id)), "tenant_id": tid,
                             "user_id": str(user_id), "request_id": request_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        state: dict[str, Any] = {}
        captured: dict[str, Any] = {}
        yield {"event": "agent_started", "data": {"conversation_id": str(conversation.id), "request_id": request_id}}
        async for mode, chunk in self.container.agent.astream(
            initial_turn_state(message, tid, str(conversation.id), request_id),
            config=config,
            stream_mode=["updates", "custom"],
            durability="exit",
        ):
            if mode == "custom":
                yield chunk
                continue
            for node, update in (chunk or {}).items():
                if not isinstance(update, dict):
                    continue
                state.update(update)
                if node in {"vector_search", "graph_search", "hybrid_search"}:
                    for key in _CAPTURE:  # a retry replaces the previous attempt's evidence
                        captured[key] = update.get(key)

        latency_ms = int((time.perf_counter() - started) * 1000)
        response = self._build_response(state, captured, latency_ms, usage.as_dict())
        assistant = Message(
            tenant_id=tenant_id, conversation_id=conversation.id, role="assistant", content=response["answer"],
            retrieval_strategy=response["retrieval_strategy"], confidence=response["confidence"], latency_ms=latency_ms,
            sources=response["sources"],
            trace={"steps": response["trace"], "verification": response["verification"],
                   "graph_evidence": response["graph_evidence"][:30],
                   "retrieved_chunks": [{k: v for k, v in c.items() if k != "text"} | {"text": truncate(c.get("text", ""), 400)}
                                        for c in response["retrieved_chunks"][:10]],
                   "intent": response["intent"], "entities": response["entities"],
                   "rewritten_query": response["rewritten_query"], "token_usage": response["token_usage"]},
        )
        self.db.add(assistant)
        conversation.updated_at = utcnow()
        await self.db.commit()
        response["conversation_id"] = str(conversation.id)
        response["message_id"] = str(assistant.id)
        logger.info(
            "chat_turn_completed",
            extra={
                "conversation_id": str(conversation.id), "question": truncate(message, 200),
                "selected_strategy": response["retrieval_strategy"], "tools_called": state.get("tools_called", []),
                "retrieval_latency": state.get("retrieval_latency_ms", 0), "llm_latency": usage.as_dict()["llm_latency_ms"],
                "total_latency": latency_ms, "token_usage": usage.total_tokens, "retry_count": response["retry_count"],
                "confidence": response["confidence"],
            },
        )
        yield {"event": "completed", "data": response}

    async def run_turn(self, tenant_id: uuid.UUID, user_id: uuid.UUID, conversation: Conversation,
                       message: str) -> dict[str, Any]:
        final: dict[str, Any] = {}
        async for event in self.stream_turn(tenant_id, user_id, conversation, message):
            if event.get("event") == "completed":
                final = event["data"]
        return final

    @staticmethod
    def _build_response(state: dict[str, Any], captured: dict[str, Any], latency_ms: int,
                        usage: dict[str, int]) -> dict[str, Any]:
        strategies = state.get("attempted_strategies") or []
        return {
            "answer": state.get("answer") or "",
            "sources": state.get("sources") or [],
            "retrieval_strategy": state.get("retrieval_strategy") if not strategies else strategies[0],
            "final_strategy": strategies[-1] if strategies else state.get("retrieval_strategy"),
            "attempted_strategies": strategies,
            "confidence": float(state.get("confidence") or 0.0),
            "intent": state.get("intent"),
            "entities": state.get("entities") or [],
            "rewritten_query": state.get("rewritten_query") or None,
            "retry_count": int(state.get("retry_count") or 0),
            "graph_evidence": captured.get("graph_results") or [],
            "retrieved_chunks": captured.get("vector_results") or [],
            "linked_entities": captured.get("linked_entities") or [],
            "answer_candidates": captured.get("answer_candidates") or [],
            "cypher": captured.get("cypher"),
            "trace": state.get("trace") or [],
            "verification": state.get("verification") or {},
            "latency_ms": latency_ms,
            "token_usage": usage,
        }
