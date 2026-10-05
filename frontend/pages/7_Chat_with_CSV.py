"""Chat with CSV: upload a table and ask analytical questions answered by validated, sandboxed SQL."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from services.api_client import APIError
from utils.session import handle_api_error, is_admin, parse_groups, require_auth, sidebar_user_box

st.set_page_config(page_title="Chat with CSV · Agentic GraphRAG", page_icon="📈", layout="wide")
client = require_auth()
sidebar_user_box()

st.title("📈 Chat with CSV")
st.caption("Upload a CSV and ask questions in plain English. Each question becomes a single read-only SQL query, "
           "validated and run in a sandbox. You can review and edit the SQL. Tables are analysed with SQL, "
           "not with document retrieval, so totals, averages and rankings are exact.")

# ------------------------------------------------------------------ upload
with st.expander("⬆️ Upload a dataset", expanded=False):
    with st.form("dataset-upload", clear_on_submit=True):
        file = st.file_uploader("CSV or TSV file with a header row", type=["csv", "tsv"])
        groups_text = st.text_input("Restrict to groups (optional, comma-separated)",
                                    help="Empty = everyone in the workspace can query it.")
        submitted = st.form_submit_button("Upload", type="primary")
    if submitted and file is not None:
        try:
            created = client.upload_dataset(file.name, file.getvalue(), parse_groups(groups_text))
        except APIError as exc:
            handle_api_error(exc)
        else:
            st.session_state["dataset_id"] = created["id"]
            st.session_state.pop("csv_history", None)
            st.success(f"Uploaded **{created['name']}**: {created['row_count']:,} rows, {len(created['columns'])} columns.")

try:
    datasets = client.datasets()
except APIError as exc:
    handle_api_error(exc)
    st.stop()

if not datasets:
    st.info("No datasets yet. Upload a CSV above (try `backend/data/datasets/employees.csv`).")
    st.stop()

by_id = {d["id"]: d for d in datasets}
default = st.session_state.get("dataset_id")
selected = st.selectbox("Dataset", list(by_id), index=list(by_id).index(default) if default in by_id else 0,
                        format_func=lambda i: f"{by_id[i]['name']} · {by_id[i]['row_count']:,} rows")
if selected != st.session_state.get("dataset_id"):
    st.session_state["dataset_id"] = selected
    st.session_state["csv_history"] = []


@st.cache_data(ttl=60, show_spinner=False)
def _detail(dataset_id: str, _token: str) -> dict[str, Any]:
    return client.dataset(dataset_id)


try:
    detail = _detail(selected, client.tokens.access_token[-12:] if client.tokens else "")
except APIError as exc:
    handle_api_error(exc)
    st.stop()

schema_tab, preview_tab = st.tabs(["Columns", "Preview"])
with schema_tab:
    rows = []
    for c in detail["columns"]:
        summary = ""
        if c["kind"] in {"number", "date"}:
            summary = f"{c.get('min')} … {c.get('max')}"
        elif c.get("top_values"):
            summary = ", ".join(c["top_values"][:6])
        rows.append({"column": c["name"], "original": c["original_name"], "type": c["kind"],
                     "distinct": c["distinct"], "nulls": c["nulls"], "values / range": summary})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
with preview_tab:
    st.dataframe(pd.DataFrame(detail["preview_rows"], columns=detail["preview_columns"]), hide_index=True,
                 width="stretch")

can_delete = is_admin()
if can_delete and st.button("🗑️ Delete this dataset"):
    try:
        client.delete_dataset(selected)
    except APIError as exc:
        handle_api_error(exc)
    else:
        st.session_state.pop("dataset_id", None)
        _detail.clear()
        st.rerun()

st.divider()


def render_result(result: dict[str, Any], key: str) -> None:
    st.markdown(result["answer"])
    frame = pd.DataFrame(result["rows"], columns=result["columns"])
    chart = result.get("chart")
    if chart and not frame.empty and chart["x"] in frame and chart["y"] in frame:
        data = frame.set_index(chart["x"])[[chart["y"]]]
        if chart["type"] == "line":
            st.line_chart(data)
        else:
            st.bar_chart(data)
    if not frame.empty:
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Download result (CSV)", frame.to_csv(index=False).encode(), file_name="result.csv",
                           mime="text/csv", key=f"dl-{key}")
    if result.get("truncated"):
        st.caption(f"Showing the first {len(result['rows'])} rows.")
    with st.expander(f"SQL · {result['planner']} planner · {result['latency_ms']} ms"):
        st.code(result["sql"], language="sql")
        st.caption(result.get("explanation") or "")
        edited = st.text_area("Edit and re-run (single SELECT over table `data`)", result["sql"], key=f"sql-{key}",
                              height=120)
        if st.button("Run edited SQL", key=f"run-{key}"):
            st.session_state["pending_sql"] = (result["question"], edited)
            st.rerun()


history: list[dict[str, Any]] = st.session_state.setdefault("csv_history", [])
for i, item in enumerate(history):
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        render_result(item, str(i))

st.caption("Try: *How many employees are in each department?* · *Average annual salary by city* · "
           "*Who has the highest salary?* · *How many people were hired per year?*")
pending = st.session_state.pop("pending_sql", None)
question = st.chat_input("Ask a question about this dataset…", max_chars=2000)
if question or pending:
    q, sql = (question, None) if question else pending
    with st.chat_message("user"):
        st.markdown(q if not sql else f"{q}\n\n_(edited SQL)_")
    with st.chat_message("assistant"):
        try:
            with st.spinner("Planning and running the query…"):
                result = client.query_dataset(selected, q, sql)
        except APIError as exc:
            if exc.code in {"QUESTION_NOT_UNDERSTOOD", "SQL_REJECTED"}:
                st.warning(exc.message)
            else:
                handle_api_error(exc)
        else:
            history.append(result)
            render_result(result, str(len(history) - 1))
