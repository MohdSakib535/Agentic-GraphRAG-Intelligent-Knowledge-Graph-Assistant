"""Chat endpoints: synchronous and Server-Sent-Events streaming."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response, status
from sse_starlette.sse import EventSourceResponse

from app.core.dependencies import (
    AccessScopeDep,
    ContainerDep,
    CurrentUser,
    CurrentUserDep,
    DBSession,
    RateLimit,
    require_admin,
)
from app.core.errors import AppError, error_payload
from app.core.logging import get_logger, request_id_ctx
from app.schemas.chat import ChatRequest, ChatResponse, ConversationOut, FeedbackRequest, MessageOut
from app.schemas.common import ERROR_RESPONSES
from app.services import export_service
from app.services.chat_service import ChatService
from app.services.feedback_service import FeedbackService

router = APIRouter(prefix="/chat", tags=["Chat"], responses=ERROR_RESPONSES)
logger = get_logger(__name__)


@router.post("", response_model=ChatResponse, dependencies=[Depends(RateLimit("chat"))],
             summary="Ask a question (agentic GraphRAG)")
async def chat(body: ChatRequest, user: CurrentUserDep, scope: AccessScopeDep, db: DBSession,
               container: ContainerDep) -> ChatResponse:
    """Runs the LangGraph agent: analyze -> route -> retrieve -> grade -> (rewrite) -> generate -> verify."""
    service = ChatService(db, container)
    conversation = await service.get_or_create_conversation(user.tenant_id, user.id, body.conversation_id, body.message)
    result = await service.run_turn(user.tenant_id, user.id, conversation, body.message, scope)
    return ChatResponse.model_validate(result)


@router.post("/stream", dependencies=[Depends(RateLimit("chat"))], summary="Ask a question with SSE streaming",
             response_description="text/event-stream with agent_started, query_analyzed, retrieval_started, "
                                  "retrieval_completed, reasoning, token, citation, verification, completed events")
async def chat_stream(body: ChatRequest, user: CurrentUserDep, scope: AccessScopeDep, db: DBSession,
                      container: ContainerDep) -> EventSourceResponse:
    service = ChatService(db, container)
    conversation = await service.get_or_create_conversation(user.tenant_id, user.id, body.conversation_id, body.message)
    request_id = request_id_ctx.get()

    async def events() -> AsyncIterator[dict[str, Any]]:
        try:
            async for event in service.stream_turn(user.tenant_id, user.id, conversation, body.message, scope):
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
    ratings = await FeedbackService(db).ratings_for(user.id, [m.id for m in messages if m.role == "assistant"])
    out = []
    for m in messages:
        item = MessageOut.model_validate(m)
        item.feedback = ratings.get(m.id)
        out.append(item)
    return out


@router.get("/conversations/{conversation_id}/export", summary="Export a conversation as Markdown or PDF",
            response_class=Response, responses={200: {"content": {"text/markdown": {}, "application/pdf": {}}}})
async def export_conversation(conversation_id: uuid.UUID, user: CurrentUserDep, db: DBSession, container: ContainerDep,
                              format: Annotated[str, Query(pattern="^(markdown|pdf)$")] = "markdown") -> Response:
    """Questions, answers, strategy/confidence and every citation, ready to share."""
    service = ChatService(db, container)
    conversation = await service.get_or_create_conversation(user.tenant_id, user.id, conversation_id, "")
    messages = await service.messages(user.tenant_id, user.id, conversation_id)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", conversation.title)[:60].strip("-") or "conversation"
    if format == "pdf":
        body = await asyncio.to_thread(export_service.to_pdf, conversation, messages)
        return Response(body, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.pdf"'})
    return Response(export_service.to_markdown(conversation, messages), media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.md"'})


@router.put("/messages/{message_id}/feedback", response_model=dict, summary="Rate an answer (👍 / 👎)")
async def rate_message(message_id: uuid.UUID, body: FeedbackRequest, user: CurrentUserDep, db: DBSession) -> dict[str, object]:
    feedback = await FeedbackService(db).rate(user.tenant_id, user.id, message_id, body.rating, body.comment)
    return {"message_id": str(message_id), "rating": feedback.rating, "comment": feedback.comment}


@router.delete("/messages/{message_id}/feedback", status_code=status.HTTP_204_NO_CONTENT, summary="Remove my rating")
async def clear_feedback(message_id: uuid.UUID, user: CurrentUserDep, db: DBSession) -> None:
    await FeedbackService(db).clear(user.tenant_id, user.id, message_id)


@router.get("/feedback", summary="Answer feedback across the tenant (admin)")
async def list_feedback(admin: Annotated[CurrentUser, Depends(require_admin)], db: DBSession,
                        rating: Annotated[int | None, Query(ge=-1, le=1)] = None,
                        limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[dict[str, object]]:
    return await FeedbackService(db).list(admin.tenant_id, rating, limit)


@router.get("/feedback/evaluation-questions", summary="Low-rated answers as evaluation questions (admin)")
async def feedback_to_evaluation(admin: Annotated[CurrentUser, Depends(require_admin)], db: DBSession) -> dict[str, object]:
    """Returns 👎-rated questions in the evaluation dataset format so they can be curated into the benchmark."""
    return await FeedbackService(db).as_evaluation_questions(admin.tenant_id)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a conversation and its agent memory")
async def delete_conversation(conversation_id: uuid.UUID, user: CurrentUserDep, db: DBSession,
                              container: ContainerDep) -> None:
    await ChatService(db, container).delete_conversation(user.tenant_id, user.id, conversation_id)
