"""Knowledge-graph explorer: interactive visualisation, entity search and details."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.graph import TYPE_COLORS, render_graph
from services.api_client import APIError
from utils.session import handle_api_error, is_admin, require_auth, sidebar_user_box

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
            st.dataframe(pd.DataFrame([{"source": r["source"], "relationship": r["relationship"], "target": r["target"],
                                        "origin": "✍️ manual" if r.get("manual") else "📄 extracted"}
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

    if is_admin():
        st.divider()
        st.subheader("✏️ Curate")
        st.caption("Fix extraction mistakes. Edits are audited, apply immediately to search and chat, and survive "
                   "document re-processing (renamed and merged names map onto this entity).")

        def done(message: str) -> None:
            st.toast(message)
            st.rerun()

        def pick_entity(label: str, key: str, exclude: str) -> dict | None:
            q = st.text_input(f"Find {label}", key=f"{key}-q", placeholder="type a name")
            try:
                found = [e for e in client.graph_entities(q=q or None, limit=50) if e["id"] != exclude] if q else []
            except APIError as exc:
                handle_api_error(exc)
                found = []
            opts = {f"{e['name']} ({e['type']})": e for e in found}
            choice = st.selectbox(label.capitalize(), ["—", *opts], key=f"{key}-sel")
            return opts.get(choice)

        edit_tab, merge_tab, rel_tab, delete_tab = st.tabs(["Edit", "Merge duplicates", "Relationships", "Delete"])
        with edit_tab, st.form(f"edit-{ent['id']}"):
            name = st.text_input("Name", ent["name"])
            etype = st.selectbox("Type", TYPES, index=TYPES.index(ent["type"]) if ent["type"] in TYPES else 0)
            description = st.text_area("Description", ent.get("description") or "")
            aliases = st.text_input("Aliases (comma-separated)", ", ".join(ent.get("aliases") or []))
            if st.form_submit_button("Save changes", type="primary"):
                try:
                    client.update_entity(ent["id"], name=name.strip(), type=etype, description=description,
                                         aliases=[a.strip() for a in aliases.split(",") if a.strip()])
                except APIError as exc:
                    st.error(str(exc))
                else:
                    done("Entity updated")
        with merge_tab:
            st.caption(f"Fold a duplicate into **{ent['name']}**: its relationships, mentions and aliases move here.")
            dup = pick_entity("duplicate", f"merge-{ent['id']}", ent["id"])
            if dup and st.button(f"Merge '{dup['name']}' into '{ent['name']}'", type="primary"):
                try:
                    client.merge_entities(ent["id"], [dup["id"]])
                except APIError as exc:
                    st.error(str(exc))
                else:
                    done("Entities merged")
        with rel_tab:
            try:
                schema = client.graph_schema()
            except APIError:
                schema = {}
            cols = st.columns([1, 2])
            direction = cols[0].radio("Direction", ["outgoing", "incoming"], key=f"dir-{ent['id']}")
            rel_type = cols[1].selectbox("Relationship", list(schema), key=f"rt-{ent['id']}",
                                         help="\n".join(schema.values()))
            other = pick_entity("other entity", f"rel-{ent['id']}", ent["id"])
            evidence = st.text_input("Evidence / note (optional)", key=f"ev-{ent['id']}")
            if other and st.button("Add relationship", type="primary"):
                src, dst = (ent, other) if direction == "outgoing" else (other, ent)
                try:
                    client.add_relationship(src["id"], rel_type, dst["id"], evidence)
                except APIError as exc:
                    st.error(str(exc))
                else:
                    done("Relationship added")
            rels = [r for r in detail["relationships"] if r.get("source_id") and r.get("target_id")]
            if rels:
                labels = {f"{r['source']} -[{r['relationship']}]-> {r['target']}": r for r in rels}
                victim = st.selectbox("Remove a relationship", ["—", *labels], key=f"rm-{ent['id']}")
                if victim != "—" and st.button("Remove relationship"):
                    r = labels[victim]
                    try:
                        client.delete_relationship(r["source_id"], r["relationship"], r["target_id"])
                    except APIError as exc:
                        st.error(str(exc))
                    else:
                        done("Relationship removed")
        with delete_tab:
            st.warning("Deleting removes the entity and all its relationships. Re-processing a document that "
                       "mentions it will create it again.")
            if st.checkbox("I understand", key=f"del-ok-{ent['id']}") and st.button("Delete entity", type="primary"):
                try:
                    client.delete_entity(ent["id"])
                except APIError as exc:
                    st.error(str(exc))
                else:
                    done("Entity deleted")
