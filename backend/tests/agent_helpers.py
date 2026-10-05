"""Helpers for driving the LangGraph agent in tests."""

from __future__ import annotations

import uuid
from typing import Any

from app.agents.workflow import RECURSION_LIMIT, initial_turn_state, thread_id
from conftest import TENANT_A


async def ask(container, question: str, tenant: str = TENANT_A, conversation: str | None = None,
              denied: tuple[str, ...] = ()) -> dict[str, Any]:
    conversation = conversation or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id(tenant, conversation), "tenant_id": tenant, "denied_document_ids": list(denied)},
              "recursion_limit": RECURSION_LIMIT}
    state: dict[str, Any] = {"events": []}
    async for mode, chunk in container.agent.astream(initial_turn_state(question, tenant, conversation, "test"),
                                                     config=config, stream_mode=["updates", "custom"]):
        if mode == "custom":
            state["events"].append(chunk["event"])
            continue
        for node, update in chunk.items():
            if node.endswith("_search"):
                state["retrieved"] = update
            if node != "finalize":
                state.update(update)
    return state
