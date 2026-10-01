"""Chat endpoints: synchronous and Server-Sent-Events streaming."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, status
from sse_starlette.sse import EventSourceResponse

from app.core.dependencies import ContainerDep, CurrentUserDep, DBSession, RateLimit
from app.core.errors import AppError, error_payload
from app.core.logging import get_logger, request_id_ctx
from app.schemas.chat import ChatRequest, ChatResponse, ConversationOut, MessageOut
from app.schemas.common import ERROR_RESPONSES
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"], responses=ERROR_RESPONSES)
logger = get_logger(__name__)


@router.post("", response_model=ChatResponse, dependencies=[Depends(RateLimit("chat"))],
             summary="Ask a question (agentic GraphRAG)")
async def chat(body: ChatRequest, user: CurrentUserDep, db: DBSession, container: ContainerDep) -> ChatResponse:
    """Runs the LangGraph agent: analyze -> route -> retrieve -> grade -> (rewrite) -> generate -> verify."""
    service = ChatService(db, container)
    conversation = await service.get_or_create_conversation(user.tenant_id, user.id, body.conversation_id, body.message)
    result = await service.run_turn(user.tenant_id, user.id, conversation, body.message)
    return ChatResponse.model_validate(result)


@router.post("/stream", dependencies=[Depends(RateLimit("chat"))], summary="Ask a question with SSE streaming",
             response_description="text/event-stream with agent_started, query_analyzed, retrieval_started, "
                                  "retrieval_completed, reasoning, token, citation, verification, completed events")
async def chat_stream(body: ChatRequest, user: CurrentUserDep, db: DBSession, container: ContainerDep) -> EventSourceResponse:
    service = ChatService(db, container)
    conversation = await service.get_or_create_conversation(user.tenant_id, user.id, body.conversation_id, body.message)
    request_id = request_id_ctx.get()

    async def events() -> AsyncIterator[dict[str, Any]]:
        try:
            async for event in service.stream_turn(user.tenant_id, user.id, conversation, body.message):
                yield {"event": event["event"], "data": json.dumps(event["data"], default=str)}
        except AppError as exc:
            yield {"event": "error", "data": json.dumps(error_payload(exc.code, exc.message, request_id))}
        except Exception:
            logger.exception("chat_stream_failed")
            yield {"event": "error",
                   "data": json.dumps(error_payload("INTERNAL_ERROR", "An unexpected error occurred", request_id))}

    return EventSourceResponse(events(), ping=15, headers={"X-Accel-Buffering": "no"})


@router.get("/conversations", response_model=list[ConversationOut], summary="List my conversations")
async def list_conversations(user: CurrentUserDep, db: DBSession, container: ContainerDep) -> list[ConversationOut]:
    convs = await ChatService(db, container).list_conversations(user.tenant_id, user.id)
    return [ConversationOut.model_validate(c) for c in convs]


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut], summary="Conversation messages")
async def conversation_messages(conversation_id: uuid.UUID, user: CurrentUserDep, db: DBSession,
                                container: ContainerDep) -> list[MessageOut]:
    messages = await ChatService(db, container).messages(user.tenant_id, user.id, conversation_id)
    return [MessageOut.model_validate(m) for m in messages]


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a conversation and its agent memory")
async def delete_conversation(conversation_id: uuid.UUID, user: CurrentUserDep, db: DBSession,
                              container: ContainerDep) -> None:
    await ChatService(db, container).delete_conversation(user.tenant_id, user.id, conversation_id)
