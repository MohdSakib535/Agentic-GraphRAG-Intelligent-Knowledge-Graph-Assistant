"""Dashboard: document, graph and conversation metrics + recent activity."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.api_client import APIError
from utils.session import handle_api_error, require_auth, sidebar_user_box

st.set_page_config(page_title="Dashboard · Agentic GraphRAG", page_icon="📊", layout="wide")
client = require_auth()
sidebar_user_box()
st.title("📊 Dashboard")

try:
    doc_stats = client.document_stats()
    graph_stats = client.graph_stats()
    conversations = client.conversations()
    recent_docs = client.list_documents(limit=10)["items"]
except APIError as exc:
    handle_api_error(exc)
    st.stop()

row1 = st.columns(4)
row1[0].metric("Total documents", doc_stats["total_documents"], border=True)
row1[1].metric("Processed", doc_stats["processed_documents"], border=True)
row1[2].metric("Processing", doc_stats["processing_documents"] + doc_stats["pending_documents"], border=True)
row1[3].metric("Failed", doc_stats["failed_documents"], border=True)
row2 = st.columns(4)
row2[0].metric("Graph entities", graph_stats["entities"], border=True)
row2[1].metric("Relationships", graph_stats["relationships"], border=True)
row2[2].metric("Chunks", graph_stats["chunks"], border=True)
row2[3].metric("Conversations", len(conversations), border=True)

left, right = st.columns(2)
with left:
    st.subheader("Entities by type")
    if graph_stats["entities_by_type"]:
        st.bar_chart(pd.DataFrame({"count": graph_stats["entities_by_type"]}), horizontal=True)
    else:
        st.info("No entities yet — upload documents to build the knowledge graph.")
with right:
    st.subheader("Relationships by type")
    if graph_stats["relationships_by_type"]:
        st.bar_chart(pd.DataFrame({"count": graph_stats["relationships_by_type"]}), horizontal=True)
    else:
        st.info("No relationships yet.")

st.subheader("Recent activity")
activity = [{"when": d["updated_at"][:19].replace("T", " "), "type": "document", "item": d["filename"],
             "status": d["status"]} for d in recent_docs]
activity += [{"when": c["updated_at"][:19].replace("T", " "), "type": "conversation", "item": c["title"],
              "status": "-"} for c in conversations[:10]]
if activity:
    st.dataframe(pd.DataFrame(sorted(activity, key=lambda a: a["when"], reverse=True)[:15]), hide_index=True,
                 width="stretch")
else:
    st.info("No activity yet.")
