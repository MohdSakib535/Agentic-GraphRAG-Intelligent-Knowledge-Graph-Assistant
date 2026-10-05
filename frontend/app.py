"""Agentic GraphRAG - Streamlit entry point.

Uses ``st.navigation`` so that unauthenticated visitors only see the login page;
the protected pages are registered only after a successful login.
"""

from __future__ import annotations

import streamlit as st

from services.api_client import PUBLIC_API_URL, APIError
from utils.session import get_client, is_admin, is_authenticated, set_session, sidebar_user_box

st.set_page_config(page_title="Agentic GraphRAG", page_icon="🕸️", layout="wide")


SSO_ERRORS = {
    "GOOGLE_EMAIL_UNVERIFIED": "Your Google email address is not verified.",
    "GOOGLE_DOMAIN_NOT_ALLOWED": "Your Google account's domain is not allowed on this deployment.",
    "GOOGLE_STATE_INVALID": "The sign-in request expired. Please try again.",
    "GOOGLE_LOGIN_CANCELLED": "Google sign-in was cancelled.",
    "ACCOUNT_DISABLED": "This account is disabled. Contact your workspace administrator.",
}


def handle_sso_redirect() -> None:
    """Google sign-in returns here with a single-use ``sso_code`` that is traded for tokens server-side."""
    params = st.query_params
    if "sso_error" in params:
        st.session_state["sso_error"] = SSO_ERRORS.get(params["sso_error"], "Google sign-in failed. Please try again.")
        st.query_params.clear()
    elif "sso_code" in params:
        code = params["sso_code"]
        st.query_params.clear()  # never leave the code in the address bar
        client = get_client()
        try:
            set_session(client, client.google_exchange(code))
        except APIError as exc:
            st.session_state["sso_error"] = str(exc)


def google_button() -> None:
    try:
        providers = st.session_state.get("auth_providers") or get_client().auth_providers()
        st.session_state["auth_providers"] = providers
    except APIError:
        return
    if providers.get("google"):
        st.link_button("Sign in with Google", f"{PUBLIC_API_URL}/auth/google/login", icon=":material/login:",
                       width="stretch")
        st.caption("or use your email and password")


def login_view() -> None:
    st.title("🕸️ Agentic GraphRAG")
    st.caption("Intelligent Knowledge Graph Assistant — graph + vector retrieval orchestrated by a LangGraph agent.")
    if error := st.session_state.pop("sso_error", None):
        st.error(error)
    google_button()
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
    cols[0].page_link(PAGES["csv"], label="Chat with CSV", icon="📈")
    cols[1].page_link(PAGES["dashboard"], label="Dashboard", icon="📊")
    cols[2].page_link(PAGES["evaluation"], label="Evaluation", icon="🧪")
    cols = st.columns(3)
    cols[0].page_link(PAGES["settings"], label="Settings", icon="⚙️")
    if is_admin():
        cols[1].page_link(ADMIN_PAGES["users"], label="Users & groups", icon="👥")
        cols[2].page_link(ADMIN_PAGES["connectors"], label="Connectors", icon="🔌")


PAGES = {
    "dashboard": st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊"),
    "documents": st.Page("pages/2_Documents.py", title="Documents", icon="📄"),
    "chat": st.Page("pages/3_Chat.py", title="Chat", icon="💬"),
    "csv": st.Page("pages/7_Chat_with_CSV.py", title="Chat with CSV", icon="📈"),
    "graph": st.Page("pages/4_Knowledge_Graph.py", title="Knowledge Graph", icon="🕸️"),
    "evaluation": st.Page("pages/5_Evaluation.py", title="Evaluation", icon="🧪"),
    "settings": st.Page("pages/6_Settings.py", title="Settings", icon="⚙️"),
}
ADMIN_PAGES = {
    "users": st.Page("pages/8_Users.py", title="Users & groups", icon="👥"),
    "connectors": st.Page("pages/9_Connectors.py", title="Connectors", icon="🔌"),
}

handle_sso_redirect()
if is_authenticated():
    home = st.Page(home_view, title="Home", icon="🏠", default=True)
    sections: dict[str, list] = {"Workspace": [home, *PAGES.values()]}
    if is_admin():
        sections["Administration"] = list(ADMIN_PAGES.values())
    navigation = st.navigation(sections)
else:
    navigation = st.navigation([st.Page(login_view, title="Log in", icon="🔐", default=True)])
navigation.run()
