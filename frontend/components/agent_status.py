"""Agent trace / live status rendering."""

from __future__ import annotations

from typing import Any

import streamlit as st

STEP_LABELS = {
    "analyze_query": "Query Analyzer",
    "vector_search": "Vector Search",
    "graph_search": "Graph Search",
    "hybrid_search": "Hybrid Search (Graph + Vector)",
    "grade_context": "Evidence Grading",
    "rewrite_query": "Query Rewriting",
    "generate_answer": "Answer Generation",
    "verify_answer": "Verification",
    "finalize": "Memory Update",
}


def _detail_line(step: dict[str, Any]) -> str:
    d = step.get("detail") or {}
    name = step.get("step")
    if name == "analyze_query":
        return f"Strategy: **{d.get('strategy')}** · intent `{d.get('intent')}` · entities {d.get('entities')}"
    if name and name.endswith("_search"):
        extra = f" · candidates {d.get('candidates')}" if d.get("candidates") else ""
        cached = " · cached" if d.get("cached") else ""
        return f"{d.get('chunks', 0)} chunks · {d.get('facts', 0)} graph facts{extra}{cached}"
    if name == "grade_context":
        return f"Evidence grade **{d.get('grade')}** ({'sufficient' if d.get('sufficient') else 'insufficient'}, {d.get('method')})"
    if name == "rewrite_query":
        return f"Attempt {d.get('attempt')}: `{d.get('rewritten_query')}` → {d.get('strategy')}"
    if name == "generate_answer":
        return f"{d.get('method')} · {d.get('citations', 0)} citations"
    if name == "verify_answer":
        return f"passed={d.get('passed')} · support={d.get('support')} · confidence={d.get('confidence')}"
    return ""


def render_trace(trace: list[dict[str, Any]], expanded: bool = False) -> None:
    if not trace:
        return
    with st.expander("🧭 Agent trace", expanded=expanded):
        for i, step in enumerate(trace):
            icon = "✅" if step.get("status") in ("done", None) else ("⚠️" if step.get("status") == "failed" else "❌")
            label = STEP_LABELS.get(step.get("step", ""), step.get("step"))
            latency = f" · {step['latency_ms']} ms" if step.get("latency_ms") is not None else ""
            st.markdown(f"{icon} **{label}**{latency}  \n{_detail_line(step)}")
            if i < len(trace) - 1:
                st.markdown("&nbsp;&nbsp;&nbsp;↓")


EVENT_MESSAGES = {
    "agent_started": "Agent started",
    "query_analyzed": "Analyzing query...",
    "retrieval_started": "Retrieving evidence...",
    "retrieval_completed": "Retrieval completed",
    "verification": "Verifying answer...",
}


def describe_event(event: str, data: dict[str, Any]) -> str | None:
    if event == "query_analyzed":
        return (f"🔎 Query analyzed → strategy **{data.get('retrieval_strategy')}** "
                f"(intent `{data.get('intent')}`, entities {data.get('entities')})")
    if event == "retrieval_started":
        return f"📡 {data.get('strategy', '').title()} search (attempt {data.get('attempt')}) — `{data.get('tool')}`"
    if event == "retrieval_completed":
        return (f"✅ {data.get('strategy', '').title()} search ✓ — {data.get('chunks')} chunks, {data.get('facts')} facts"
                f" in {data.get('latency_ms')} ms")
    if event == "reasoning":
        return f"💭 {data.get('message')}"
    if event == "verification":
        ok = data.get("passed")
        return f"{'🛡️' if ok else '⚠️'} Verification {'passed' if ok else 'failed'} (support {data.get('support_score')})"
    return None
