"""Authentication/session helpers backed by ``st.session_state``.

Tokens live only in server-side Streamlit session state (never in URLs, query
params or browser storage) and are cleared on logout.
"""

from __future__ import annotations

import contextlib
from typing import Any

import streamlit as st

from services.api_client import APIClient, APIError, Tokens

_AUTH_KEYS = ("tokens", "user", "tenant")
_CHAT_KEYS = ("conversation_id", "messages")


def _store_tokens(tokens: Tokens) -> None:
    st.session_state["tokens"] = tokens


def get_client() -> APIClient:
    return APIClient(tokens=st.session_state.get("tokens"), on_tokens_refreshed=_store_tokens)


def is_authenticated() -> bool:
    return st.session_state.get("tokens") is not None


def set_session(client: APIClient, auth_response: dict[str, Any]) -> None:
    st.session_state["tokens"] = client.tokens
    st.session_state["user"] = auth_response.get("user")
    try:
        st.session_state["tenant"] = client.me().get("tenant")
    except APIError:
        st.session_state["tenant"] = None


def clear_session() -> None:
    for key in (*_AUTH_KEYS, *_CHAT_KEYS):
        st.session_state.pop(key, None)


def logout() -> None:
    with contextlib.suppress(APIError):  # the local session is cleared regardless
        get_client().logout()
    clear_session()


def require_auth() -> APIClient:
    """Stop rendering the page unless the user is logged in."""
    if not is_authenticated():
        st.warning("Please log in to continue.")
        if st.button("Go to login"):
            st.rerun()
        st.stop()
    return get_client()


def sidebar_user_box() -> None:
    with st.sidebar:
        user = st.session_state.get("user") or {}
        tenant = st.session_state.get("tenant") or {}
        if user:
            st.caption("Signed in as")
            st.markdown(f"**{user.get('full_name') or user.get('email')}**")
            if tenant:
                st.caption(f"Workspace: {tenant.get('name')}")
            if st.button("Log out", width="stretch"):
                logout()
                st.rerun()


def handle_api_error(exc: APIError) -> None:
    if exc.status == 401:
        clear_session()
        st.error("Your session expired. Please log in again.")
        if st.button("Log in again"):
            st.rerun()
        st.stop()
    elif exc.status == 429:
        st.warning(f"Rate limit reached: {exc.message}")
    else:
        st.error(f"{exc.code}: {exc}")
