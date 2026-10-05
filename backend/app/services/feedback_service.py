"""Answer feedback (thumbs up/down) and conversion of low-rated answers into evaluation questions."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.metrics import FEEDBACK
from app.models.conversation import Conversation
from app.models.feedback import MessageFeedback
from app.models.message import Message
from app.models.user import User
from app.services.audit import record_audit


class FeedbackService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _owned_assistant_message(self, tenant_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID) -> Message:
        row = (
            await self.db.execute(
                select(Message).join(Conversation, Conversation.id == Message.conversation_id).where(
                    Message.id == message_id, Message.tenant_id == tenant_id, Message.role == "assistant",
                    Conversation.user_id == user_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise NotFoundError("Message not found", code="MESSAGE_NOT_FOUND")
        return row

    async def rate(self, tenant_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID, rating: int,
                   comment: str | None) -> MessageFeedback:
        await self._owned_assistant_message(tenant_id, user_id, message_id)
        feedback = (
            await self.db.execute(select(MessageFeedback).where(MessageFeedback.message_id == message_id,
                                                                MessageFeedback.user_id == user_id))
        ).scalar_one_or_none()
        if feedback is None:
            feedback = MessageFeedback(tenant_id=tenant_id, message_id=message_id, user_id=user_id, rating=rating,
                                       comment=comment)
            self.db.add(feedback)
        else:
            feedback.rating, feedback.comment = rating, comment
        record_audit(self.db, "chat.feedback", tenant_id=tenant_id, user_id=user_id, resource_type="message",
                     resource_id=str(message_id), details={"rating": rating})
        await self.db.commit()
        FEEDBACK.labels(rating="up" if rating > 0 else "down").inc()
        return feedback

    async def clear(self, tenant_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID) -> None:
        await self._owned_assistant_message(tenant_id, user_id, message_id)
        feedback = (
            await self.db.execute(select(MessageFeedback).where(MessageFeedback.message_id == message_id,
                                                                MessageFeedback.user_id == user_id))
        ).scalar_one_or_none()
        if feedback is not None:
            await self.db.delete(feedback)
            await self.db.commit()

    async def ratings_for(self, user_id: uuid.UUID, message_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
        if not message_ids:
            return {}
        rows = await self.db.execute(select(MessageFeedback.message_id, MessageFeedback.rating).where(
            MessageFeedback.user_id == user_id, MessageFeedback.message_id.in_(message_ids)))
        return {m: r for m, r in rows.all()}

    async def list(self, tenant_id: uuid.UUID, rating: int | None, limit: int) -> list[dict[str, Any]]:
        """Feedback with the question that produced each rated answer (admin view)."""
        query = (
            select(MessageFeedback, Message, User.email)
            .join(Message, and_(Message.id == MessageFeedback.message_id, Message.tenant_id == tenant_id))
            .join(User, User.id == MessageFeedback.user_id)
            .where(MessageFeedback.tenant_id == tenant_id)
            .order_by(MessageFeedback.created_at.desc())
            .limit(limit)
        )
        if rating is not None:
            query = query.where(MessageFeedback.rating == rating)
        out = []
        for feedback, message, email in (await self.db.execute(query)).all():
            question = (
                await self.db.execute(
                    select(Message.content).where(Message.conversation_id == message.conversation_id,
                                                  Message.role == "user", Message.created_at <= message.created_at)
                    .order_by(Message.created_at.desc()).limit(1)
                )
            ).scalar_one_or_none()
            out.append({
                "id": str(feedback.id), "message_id": str(message.id), "rating": feedback.rating,
                "comment": feedback.comment, "user": email, "question": question, "answer": message.content,
                "retrieval_strategy": message.retrieval_strategy, "confidence": message.confidence,
                "created_at": feedback.created_at.isoformat(),
            })
        return out

    async def as_evaluation_questions(self, tenant_id: uuid.UUID, limit: int = 200) -> dict[str, Any]:
        """Low-rated answers in the evaluation-dataset format; reviewers fill in expected_keywords."""
        items = await self.list(tenant_id, rating=-1, limit=limit)
        seen: set[str] = set()
        questions = []
        for item in items:
            q = (item["question"] or "").strip()
            if not q or q.lower() in seen:
                continue
            seen.add(q.lower())
            questions.append({
                "id": f"FB{len(questions) + 1:03d}", "category": "user_feedback", "question": q,
                "expected_strategy": None, "expected_keywords": [],
                "notes": {"rejected_answer": item["answer"][:500], "comment": item["comment"],
                          "strategy_used": item["retrieval_strategy"]},
            })
        return {"description": "Questions whose answers users rated 👎. Fill in expected_keywords (or mark them "
                               "unanswerable) and append to data/evaluation/questions.json.", "questions": questions}
