"""Knowledge-graph explorer: interactive visualisation, entity search and details."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.graph import TYPE_COLORS, render_graph
from services.api_client import APIError
from utils.session import handle_api_error, require_auth, sidebar_user_box

st.set_page_config(page_title="Knowledge Graph · Agentic GraphRAG", page_icon="🕸️", layout="wide")
client = require_auth()
sidebar_user_box()
st.title("🕸️ Knowledge Graph")

TYPES = list(TYPE_COLORS)
try:
    stats = client.graph_stats()
except APIError as exc:
    handle_api_error(exc)
    st.stop()

cols = st.columns(4)
cols[0].metric("Entities", stats["entities"])
cols[1].metric("Relationships", stats["relationships"])
cols[2].metric("Chunks", stats["chunks"])
cols[3].metric("Documents", stats["documents"])
st.markdown(" ".join(f"<span style='color:{c}'>●</span> {t}" for t, c in TYPE_COLORS.items()), unsafe_allow_html=True)

left, right = st.columns([1, 2])
with left:
    query = st.text_input("Search entity", placeholder="e.g. Rahul, Kafka, Project Alpha")
    type_filter = st.multiselect("Entity types", TYPES, default=[])
    try:
        entities = client.graph_entities(q=query or None, types=type_filter or None, limit=200)
    except APIError as exc:
        handle_api_error(exc)
        entities = []
    options = {f"{e['name']} ({e['type']}) · {e['degree']} links": e for e in entities}
    selected_label = st.selectbox("Select entity", ["— whole graph —", *options])
    selected = options.get(selected_label)
    depth = st.slider("Neighbourhood depth", 1, 3, 1, disabled=selected is None)
    limit = st.slider("Max relationships", 50, 1000, 300, step=50)

with right:
    try:
        sub = client.graph_subgraph(entity_id=selected["id"] if selected else None,
                                    types=type_filter or None if not selected else None, limit=limit, depth=depth)
    except APIError as exc:
        handle_api_error(exc)
        st.stop()
    if sub["nodes"]:
        render_graph(sub["nodes"], sub["edges"], highlight=selected["id"] if selected else None)
    else:
        st.info("No graph data yet — upload documents first.")

if selected:
    try:
        detail = client.graph_entity(selected["id"])
    except APIError as exc:
        handle_api_error(exc)
        st.stop()
    ent = detail["entity"]
    st.subheader(f"{ent['name']} · {ent['type']}")
    if ent.get("description"):
        st.write(ent["description"])
    if ent.get("aliases"):
        st.caption("Aliases: " + ", ".join(ent["aliases"]))
    a, b = st.columns(2)
    with a:
        st.markdown("**Relationships**")
        if detail["relationships"]:
            st.dataframe(pd.DataFrame([{"source": r["source"], "relationship": r["relationship"], "target": r["target"]}
                                       for r in detail["relationships"]]), hide_index=True, width="stretch")
        st.markdown("**Connected nodes**")
        st.dataframe(pd.DataFrame([{"name": n["name"], "type": n["type"]} for n in detail["neighbors"]]),
                     hide_index=True, width="stretch")
    with b:
        st.markdown("**Source documents**")
        for src in detail["sources"]:
            loc = src.get("source_filename") or "document"
            if src.get("page_number"):
                loc += f", page {src['page_number']}"
            with st.expander(loc):
                st.caption(src.get("snippet") or "")
