"""Workspace administration: users, roles, access groups and answer feedback."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from services.api_client import APIError
from utils.session import handle_api_error, parse_groups, require_admin, sidebar_user_box

st.set_page_config(page_title="Users · Agentic GraphRAG", page_icon="👥", layout="wide")
client = require_admin()
sidebar_user_box()

st.title("👥 Users & groups")
st.caption("Groups control document-level permissions: a document restricted to groups is visible only to "
           "members of at least one of them (admins see everything). Unrestricted documents are visible to all.")

users_tab, feedback_tab = st.tabs(["Users", "Answer feedback"])

with users_tab:
    try:
        users = client.list_users()
    except APIError as exc:
        handle_api_error(exc)
        users = []
    me = (st.session_state.get("user") or {}).get("id")
    if users:
        st.dataframe(pd.DataFrame([{
            "email": u["email"], "name": u.get("full_name") or "", "role": u["role"],
            "groups": ", ".join(u.get("groups") or []), "sign-in": u.get("auth_provider", "password"),
            "active": u["is_active"],
        } for u in users]), hide_index=True, width="stretch")

        st.subheader("Edit a user")
        by_id = {u["id"]: u for u in users}
        uid = st.selectbox("User", list(by_id), format_func=lambda i: by_id[i]["email"])
        target = by_id[uid]
        with st.form(f"edit-{uid}"):
            role = st.selectbox("Role", ["member", "admin"], index=0 if target["role"] == "member" else 1,
                                disabled=uid == me)
            groups = st.text_input("Groups (comma-separated)", ", ".join(target.get("groups") or []))
            active = st.checkbox("Active", value=target["is_active"], disabled=uid == me,
                                 help="Deactivating signs the user out everywhere.")
            if st.form_submit_button("Save", type="primary"):
                changes = {"groups": parse_groups(groups)}
                if uid != me:
                    changes.update(role=role, is_active=active)
                try:
                    client.update_user(uid, **changes)
                except APIError as exc:
                    handle_api_error(exc)
                else:
                    st.success("Saved.")
                    st.rerun()

    st.subheader("Add a user")
    with st.form("create-user", clear_on_submit=True):
        cols = st.columns(2)
        email = cols[0].text_input("Email")
        full_name = cols[1].text_input("Full name")
        password = cols[0].text_input("Initial password", type="password",
                                      help="At least 8 characters with upper- and lower-case letters and a digit.")
        role = cols[1].selectbox("Role", ["member", "admin"])
        groups = st.text_input("Groups (comma-separated)", placeholder="engineering, hr")
        if st.form_submit_button("Create user", type="primary"):
            try:
                client.create_user(email.strip(), password, role, parse_groups(groups), full_name.strip())
            except APIError as exc:
                handle_api_error(exc)
            else:
                st.success(f"Created {email}.")
                st.rerun()

with feedback_tab:
    st.caption("👍 / 👎 ratings from the Chat page. Down-voted questions can be exported in the evaluation-dataset "
               "format and added to the benchmark.")
    rating = st.radio("Show", ["All", "👎 only", "👍 only"], horizontal=True)
    try:
        rows = client.feedback({"All": None, "👎 only": -1, "👍 only": 1}[rating])
    except APIError as exc:
        handle_api_error(exc)
        rows = []
    if rows:
        st.dataframe(pd.DataFrame([{
            "rating": "👍" if r.get("rating", 0) > 0 else "👎", "question": r.get("question"),
            "answer": (r.get("answer") or "")[:200], "comment": r.get("comment") or "",
            "strategy": r.get("retrieval_strategy"), "when": r.get("created_at"),
        } for r in rows]), hide_index=True, width="stretch")
    else:
        st.info("No feedback yet.")
    if st.button("Export 👎 questions as evaluation items"):
        try:
            dataset = client.feedback_evaluation_questions()
        except APIError as exc:
            handle_api_error(exc)
        else:
            st.download_button("Download JSON", json.dumps(dataset, indent=2).encode(), "feedback-questions.json",
                               "application/json")
