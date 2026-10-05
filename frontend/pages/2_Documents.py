"""Documents: upload, list, ingestion progress, metadata and deletion."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from services.api_client import APIError
from utils.session import handle_api_error, is_admin, parse_groups, require_auth, sidebar_user_box

st.set_page_config(page_title="Documents · Agentic GraphRAG", page_icon="📄", layout="wide")
client = require_auth()
sidebar_user_box()
st.title("📄 Documents")

PIPELINE = [
    ("QUEUED", "Queued"), ("PARSING", "Parsing"), ("CLEANING", "Cleaning"), ("CHUNKING", "Chunking"),
    ("EXTRACTING_ENTITIES", "Entity extraction"), ("EXTRACTING_RELATIONSHIPS", "Relationship extraction"),
    ("RESOLVING_ENTITIES", "Entity resolution"), ("BUILDING_GRAPH", "Graph construction"), ("EMBEDDING", "Embedding"),
    ("INDEXING", "Vector indexing"), ("COMPLETED", "Completed"),
]
STAGES = [s for s, _ in PIPELINE]


def render_pipeline(job: dict | None) -> None:
    if not job:
        st.caption("No ingestion job.")
        return
    stage = job.get("stage")
    failed = job.get("status") == "FAILED"
    current = STAGES.index(stage) if stage in STAGES else (len(STAGES) - 1 if job.get("status") == "COMPLETED" else 0)
    st.progress(int(job.get("progress") or 0) / 100, text=f"{job.get('status')} · {stage} · {job.get('progress')}%")
    lines = []
    for i, (_, label) in enumerate(PIPELINE):
        if failed and i == current:
            icon = "❌"
        elif i < current or job.get("status") == "COMPLETED":
            icon = "✅"
        elif i == current:
            icon = "⏳"
        else:
            icon = "▫️"
        lines.append(f"{icon} {label}")
    st.markdown("  \n↓  \n".join(lines))
    if failed:
        st.error(f"{job.get('error_code')}: {job.get('error_message')}")


with st.expander("⬆️ Upload document", expanded=True):
    uploads = st.file_uploader("PDF (scanned PDFs are OCR'd), DOCX, TXT or Markdown", type=["pdf", "docx", "txt", "md"],
                               accept_multiple_files=True)
    st.caption("Spreadsheets (CSV/TSV) are analysed on the **Chat with CSV** page.")
    my_groups = (st.session_state.get("user") or {}).get("groups") or []
    groups_text = st.text_input(
        "Restrict to groups (optional, comma-separated)",
        help="Empty = visible to everyone in the workspace. Otherwise only members of these groups (and admins) "
             "can see the document, its chunks and the graph facts it supports."
             + ("" if is_admin() else f" You can use your groups: {', '.join(my_groups) or 'none'}."))
    if uploads and st.button("Upload & process", type="primary"):
        for up in uploads:
            try:
                res = client.upload_document(up.name, up.getvalue(), up.type, parse_groups(groups_text))
                st.success(f"Queued **{up.name}** (job {res['job']['id'][:8]}…)")
                st.session_state["watch_document"] = res["document"]["id"]
            except APIError as exc:
                if exc.status == 401:
                    handle_api_error(exc)
                st.error(f"{up.name}: {exc}")

watch = st.session_state.get("watch_document")
if watch:
    try:
        status = client.document_status(watch)
    except APIError:
        status = None
        st.session_state.pop("watch_document", None)
    if status:
        st.subheader("Ingestion progress")
        render_pipeline(status.get("job"))
        if status["status"] in ("PENDING", "PROCESSING"):
            time.sleep(1.5)
            st.rerun()
        elif st.button("Dismiss"):
            st.session_state.pop("watch_document", None)
            st.rerun()

st.subheader("Your documents")
status_filter = st.selectbox("Filter by status", ["all", "completed", "processing", "pending", "failed"], index=0)
try:
    docs = client.list_documents(limit=200, status=None if status_filter == "all" else status_filter)
except APIError as exc:
    handle_api_error(exc)
    st.stop()

items = docs["items"]
if not items:
    st.info("No documents yet.")
    st.stop()

table = pd.DataFrame([{
    "filename": d["filename"], "type": d["file_type"], "status": d["status"], "pages": d["page_count"],
    "chunks": d["chunk_count"], "entities": d["entity_count"], "relationships": d["relationship_count"],
    "access": ", ".join(d.get("access_groups") or []) or "everyone",
    "source": d.get("source", "upload"),
    "size_kb": round(d["size_bytes"] / 1024, 1), "uploaded": d["created_at"][:19].replace("T", " "),
} for d in items])
st.dataframe(table, hide_index=True, width="stretch")

labels = {f"{d['filename']} · {d['status']} · {d['id'][:8]}": d for d in items}
choice = st.selectbox("Select a document", list(labels))
doc = labels[choice]
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown(f"**{doc['title'] or doc['filename']}**")
with c2:
    if st.button("🔄 Reprocess"):
        try:
            client.reprocess_document(doc["id"])
            st.session_state["watch_document"] = doc["id"]
            st.rerun()
        except APIError as exc:
            st.error(str(exc))
with c3:
    confirm = st.checkbox("Confirm delete")
    if st.button("🗑️ Delete", disabled=not confirm, type="secondary"):
        try:
            client.delete_document(doc["id"])
            st.success("Document deleted.")
            st.rerun()
        except APIError as exc:
            st.error(str(exc))

if is_admin():
    with st.form(f"access-{doc['id']}"):
        new_groups = st.text_input("Access groups for this document (empty = everyone)",
                                   ", ".join(doc.get("access_groups") or []))
        if st.form_submit_button("Update access"):
            try:
                client.set_document_access(doc["id"], parse_groups(new_groups))
            except APIError as exc:
                handle_api_error(exc)
            else:
                st.success("Access updated. It applies immediately to search, the graph and chat.")
                st.rerun()

try:
    detail = client.document(doc["id"])
except APIError as exc:
    handle_api_error(exc)
    st.stop()
m, p = st.columns(2)
with m:
    st.markdown("**Metadata**")
    st.json({k: detail[k] for k in ("id", "filename", "file_type", "size_bytes", "checksum", "status", "page_count",
                                    "chunk_count", "entity_count", "relationship_count")} | {"metadata": detail["metadata"]},
            expanded=False)
with p:
    st.markdown("**Ingestion**")
    render_pipeline(detail.get("latest_job"))
    if detail.get("latest_job") and detail["latest_job"].get("stats"):
        st.json(detail["latest_job"]["stats"], expanded=False)
