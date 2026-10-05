"""Chat message rendering (answer + sources + evidence + trace)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.agent_status import render_trace
from components.citations import render_citations
from services.api_client import APIError

STRATEGY_BADGES = {"VECTOR": "🔵 VECTOR", "GRAPH": "🟠 GRAPH", "HYBRID": "🟣 HYBRID", "DIRECT": "⚪ DIRECT"}


def render_meta(result: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Strategy", STRATEGY_BADGES.get(result.get("retrieval_strategy") or "", result.get("retrieval_strategy")))
    cols[1].metric("Confidence", f"{float(result.get('confidence') or 0):.2f}")
    cols[2].metric("Retries", result.get("retry_count", 0))
    cols[3].metric("Latency", f"{result.get('latency_ms', 0)} ms")


def render_graph_evidence(facts: list[dict[str, Any]]) -> None:
    if not facts:
        return
    with st.expander(f"🕸️ Graph evidence ({len(facts)} facts)"):
        df = pd.DataFrame([{"source": f["source"], "relationship": f["relationship"], "target": f["target"],
                            "hops": f.get("hops", 1), "score": round(float(f.get("score", 0)), 2)} for f in facts])
        st.dataframe(df, hide_index=True, width="stretch")


def render_chunks(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return
    with st.expander(f"📄 Retrieved chunks ({len(chunks)})"):
        for c in chunks:
            loc = c.get("source_filename") or "document"
            if c.get("page_number"):
                loc += f", page {c['page_number']}"
            retrievers = ", ".join(c.get("retrievers") or [])
            st.markdown(f"**{loc}** · score {float(c.get('score', 0)):.3f} · _{retrievers}_")
            st.caption((c.get("text") or "")[:600])


def render_assistant_result(result: dict[str, Any], show_answer: bool = True) -> None:
    if show_answer:
        st.markdown(result.get("answer", ""))
    render_citations(result.get("sources") or [])
    render_meta(result)
    if result.get("rewritten_query"):
        st.caption(f"Rewritten query: _{result['rewritten_query']}_")
    render_graph_evidence(result.get("graph_evidence") or [])
    render_chunks(result.get("retrieved_chunks") or [])
    render_trace(result.get("trace") or [])


def render_feedback(message_id: str | None, current: int | None = None) -> None:
    """👍 / 👎 on an answer. Ratings feed the admin feedback view and the evaluation set."""
    if not message_id:
        return
    from utils.session import get_client

    key = f"fb-{message_id}"
    rating = st.session_state.get(key, current)
    cols = st.columns([1, 1, 10])
    for col, value, icon in ((cols[0], 1, "👍"), (cols[1], -1, "👎")):
        if col.button(icon, key=f"{key}-{value}", type="primary" if rating == value else "secondary",
                      help="Helpful" if value > 0 else "Not helpful"):
            try:
                if rating == value:
                    get_client().clear_rating(message_id)
                    st.session_state[key] = None
                else:
                    get_client().rate_message(message_id, value)
                    st.session_state[key] = value
            except APIError as exc:
                st.error(str(exc))
            st.rerun()
    if rating:
        cols[2].caption("Thanks for the feedback!" if rating > 0 else "Thanks — this question is flagged for review.")


def render_stored_message(message: dict[str, Any]) -> None:
    """Render a message loaded from the conversation history API."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            trace = message.get("trace") or {}
            render_assistant_result({
                "sources": message.get("sources") or [],
                "retrieval_strategy": message.get("retrieval_strategy"),
                "confidence": message.get("confidence"),
                "retry_count": sum(1 for step in trace.get("steps") or [] if step.get("step") == "rewrite_query"),
                "latency_ms": message.get("latency_ms") or 0,
                "graph_evidence": trace.get("graph_evidence") or [],
                "retrieved_chunks": trace.get("retrieved_chunks") or [],
                "trace": trace.get("steps") or [],
                "rewritten_query": trace.get("rewritten_query"),
            }, show_answer=False)
            render_feedback(message.get("id"), message.get("feedback"))
