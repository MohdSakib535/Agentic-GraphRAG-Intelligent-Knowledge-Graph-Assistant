"""Agentic GraphRAG - Streamlit entry point.

Uses ``st.navigation`` so that unauthenticated visitors only see the login page;
the protected pages are registered only after a successful login.
"""

from __future__ import annotations

import streamlit as st

from services.api_client import APIError
from utils.session import get_client, is_authenticated, set_session, sidebar_user_box

st.set_page_config(page_title="Agentic GraphRAG", page_icon="🕸️", layout="wide")


def login_view() -> None:
    st.title("🕸️ Agentic GraphRAG")
    st.caption("Intelligent Knowledge Graph Assistant — graph + vector retrieval orchestrated by a LangGraph agent.")
    login_tab, register_tab = st.tabs(["Log in", "Register"])
    with login_tab:
        with st.form("login"):
            email = st.text_input("Email", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            client = get_client()
            try:
                data = client.login(email.strip(), password)
            except APIError as exc:
                st.error(str(exc))
            else:
                set_session(client, data)
                st.rerun()
    with register_tab:
        with st.form("register"):
            full_name = st.text_input("Full name")
            r_email = st.text_input("Work email")
            tenant = st.text_input("Organisation / workspace name", help="A new isolated workspace is created for you.")
            r_password = st.text_input("Password", type="password",
                                       help="At least 8 characters with upper- and lower-case letters and a digit.")
            r_confirm = st.text_input("Confirm password", type="password")
            r_submitted = st.form_submit_button("Create account", type="primary")
        if r_submitted:
            if r_password != r_confirm:
                st.error("Passwords do not match.")
            else:
                client = get_client()
                try:
                    data = client.register(r_email.strip(), r_password, tenant.strip(), full_name.strip() or None)
                except APIError as exc:
                    st.error(str(exc))
                else:
                    set_session(client, data)
                    st.success("Account created.")
                    st.rerun()


def home_view() -> None:
    sidebar_user_box()
    user = st.session_state.get("user") or {}
    st.title("🕸️ Agentic GraphRAG")
    st.write(f"Welcome back, **{user.get('full_name') or user.get('email')}**.")
    st.markdown(
        """
        This assistant answers questions over **your** documents by combining a Neo4j knowledge graph with vector
        search. A LangGraph agent analyzes each question, chooses **VECTOR**, **GRAPH** or **HYBRID** retrieval,
        grades the evidence, rewrites the query when needed (max 3 retries), and verifies the grounded answer
        before returning it with citations.
        """
    )
    cols = st.columns(3)
    cols[0].page_link(PAGES["documents"], label="Upload documents", icon="📄")
    cols[1].page_link(PAGES["chat"], label="Ask questions", icon="💬")
    cols[2].page_link(PAGES["graph"], label="Explore the graph", icon="🕸️")
    cols = st.columns(3)
    cols[0].page_link(PAGES["dashboard"], label="Dashboard", icon="📊")
    cols[1].page_link(PAGES["evaluation"], label="Evaluation", icon="🧪")
    cols[2].page_link(PAGES["settings"], label="Settings", icon="⚙️")


PAGES = {
    "dashboard": st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊"),
    "documents": st.Page("pages/2_Documents.py", title="Documents", icon="📄"),
    "chat": st.Page("pages/3_Chat.py", title="Chat", icon="💬"),
    "graph": st.Page("pages/4_Knowledge_Graph.py", title="Knowledge Graph", icon="🕸️"),
    "evaluation": st.Page("pages/5_Evaluation.py", title="Evaluation", icon="🧪"),
    "settings": st.Page("pages/6_Settings.py", title="Settings", icon="⚙️"),
}

if is_authenticated():
    navigation = st.navigation([st.Page(home_view, title="Home", icon="🏠", default=True), *PAGES.values()])
else:
    navigation = st.navigation([st.Page(login_view, title="Log in", icon="🔐", default=True)])
navigation.run()
