"""Evaluation: compare Vector RAG vs GraphRAG vs Agentic GraphRAG."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from services.api_client import APIError
from utils.session import handle_api_error, require_auth, sidebar_user_box

st.set_page_config(page_title="Evaluation · Agentic GraphRAG", page_icon="🧪", layout="wide")
client = require_auth()
sidebar_user_box()
st.title("🧪 Evaluation")

SYSTEM_LABELS = {"vector_rag": "Vector RAG", "graph_rag": "GraphRAG", "agentic_graphrag": "Agentic GraphRAG"}

try:
    dataset = client.evaluation_dataset()
except APIError as exc:
    handle_api_error(exc)
    st.stop()
categories = sorted({q["category"] for q in dataset["questions"]})

with st.expander("▶️ Run a new evaluation", expanded=False):
    systems = st.multiselect("Systems", list(SYSTEM_LABELS), default=list(SYSTEM_LABELS),
                             format_func=SYSTEM_LABELS.get)
    cats = st.multiselect("Categories (empty = all)", categories)
    if st.button("Run evaluation", type="primary", disabled=not systems):
        try:
            run = client.run_evaluation(systems, cats or None)
            st.session_state["eval_run"] = run["id"]
            st.success(f"Evaluation run {run['id'][:8]}… queued.")
        except APIError as exc:
            handle_api_error(exc)

try:
    data = client.evaluation_results()
except APIError as exc:
    handle_api_error(exc)
    st.stop()
run = data.get("run")
if not run:
    st.info(f"No evaluation results yet. The dataset contains {dataset['total']} questions across "
            f"{len(categories)} categories: {', '.join(categories)}.")
    st.stop()
if run["status"] in ("QUEUED", "RUNNING"):
    st.info(f"Run {run['id'][:8]}… is {run['status'].lower()}…")
    time.sleep(3)
    st.rerun()
if run["status"] == "FAILED":
    st.error(f"Last run failed: {run.get('error_message')}")
    st.stop()

summary = {k: v for k, v in (run.get("summary") or {}).items() if k in SYSTEM_LABELS}
st.caption(f"Run {run['id'][:8]} · {run['question_count']} questions · finished {str(run.get('finished_at'))[:19]}")
cols = st.columns(4)
cols[0].metric("Evaluation questions", run["question_count"])
best = summary.get("agentic_graphrag") or next(iter(summary.values()), {})
cols[1].metric("Agentic accuracy", f"{best.get('accuracy', 0):.2f}")
cols[2].metric("Agentic faithfulness", f"{best.get('faithfulness', 0):.2f}")
cols[3].metric("Routing accuracy", f"{best.get('routing_accuracy') or 0:.2f}")

metrics = ["accuracy", "faithfulness", "context_relevance", "retrieval_recall"]
table = pd.DataFrame([{"system": SYSTEM_LABELS[s], **{m: v[m] for m in metrics},
                       "avg_latency_ms": v["avg_latency_ms"], "avg_token_usage": v["avg_token_usage"]}
                      for s, v in summary.items()]).set_index("system")
st.subheader("System comparison")
st.dataframe(table, width="stretch")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Quality metrics**")
    st.bar_chart(table[metrics].T, stack=False)
with c2:
    st.markdown("**Average latency (ms)**")
    st.bar_chart(table[["avg_latency_ms"]])

st.subheader("Accuracy by category")
by_cat = pd.DataFrame({SYSTEM_LABELS[s]: v.get("by_category", {}) for s, v in summary.items()})
st.dataframe(by_cat, width="stretch")
st.bar_chart(by_cat, stack=False)

st.subheader("Per-question results")
rows = pd.DataFrame(data["results"])
if not rows.empty:
    system_pick = st.selectbox("System", list(SYSTEM_LABELS), format_func=SYSTEM_LABELS.get, index=2)
    view = rows[rows["system"] == system_pick][["question_id", "category", "question", "expected_strategy",
                                                "selected_strategy", "correctness", "faithfulness",
                                                "retrieval_recall", "latency_ms", "answer"]]
    st.dataframe(view, hide_index=True, width="stretch")
