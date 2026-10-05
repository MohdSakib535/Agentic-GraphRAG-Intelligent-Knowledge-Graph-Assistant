"""ChatGPT-style chat with live agent activity (SSE), citations, evidence and trace."""

from __future__ import annotations

import streamlit as st

from components.agent_status import describe_event
from components.chat import render_assistant_result, render_feedback, render_stored_message
from services.api_client import APIError
from utils.session import handle_api_error, require_auth, sidebar_user_box

st.set_page_config(page_title="Chat · Agentic GraphRAG", page_icon="💬", layout="wide")
client = require_auth()
sidebar_user_box()

# ------------------------------------------------------------- conversations
with st.sidebar:
    st.divider()
    st.subheader("Conversations")
    if st.button("➕ New conversation", width="stretch"):
        st.session_state["conversation_id"] = None
        st.session_state["messages"] = []
        st.rerun()
    try:
        conversations = client.conversations()
    except APIError as exc:
        handle_api_error(exc)
        conversations = []
    for conv in conversations[:30]:
        active = conv["id"] == st.session_state.get("conversation_id")
        if st.button(("▶ " if active else "") + conv["title"][:40], key=f"conv-{conv['id']}", width="stretch"):
            st.session_state["conversation_id"] = conv["id"]
            st.session_state["messages"] = None  # load lazily below
            st.session_state.pop("export", None)
            st.rerun()
    if st.session_state.get("conversation_id") and st.button("🗑️ Delete conversation", width="stretch"):
        try:
            client.delete_conversation(st.session_state["conversation_id"])
        except APIError as exc:
            st.error(str(exc))
        st.session_state["conversation_id"] = None
        st.session_state["messages"] = []
        st.rerun()
    streaming = st.toggle("Stream agent activity (SSE)", value=True)
    if st.session_state.get("conversation_id"):
        st.divider()
        st.subheader("Export")
        fmt = st.radio("Format", ["Markdown", "PDF"], horizontal=True, label_visibility="collapsed")
        if st.button("Prepare export", width="stretch"):
            try:
                st.session_state["export"] = (fmt, client.export_conversation(st.session_state["conversation_id"],
                                                                              fmt.lower()))
            except APIError as exc:
                st.error(str(exc))
        if export := st.session_state.get("export"):
            ext, mime = (".pdf", "application/pdf") if export[0] == "PDF" else (".md", "text/markdown")
            st.download_button(f"Download {export[0]}", export[1], file_name=f"conversation{ext}", mime=mime,
                               width="stretch")

st.title("💬 Chat")
st.caption("Try: *What is Kafka?* · *Who manages Project Alpha?* · "
           "*Which developers work on Kafka projects managed by Rahul?* · then *What technologies does that project use?*")

conversation_id = st.session_state.get("conversation_id")
if st.session_state.get("messages") is None and conversation_id:
    try:
        st.session_state["messages"] = [
            {"role": m["role"], "stored": m} for m in client.conversation_messages(conversation_id)
        ]
    except APIError as exc:
        handle_api_error(exc)
        st.session_state["messages"] = []
messages = st.session_state.setdefault("messages", []) or []

for msg in messages:
    if "stored" in msg:
        render_stored_message(msg["stored"])
    elif msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            render_assistant_result(msg["result"])
            render_feedback(msg["result"].get("message_id"))

prompt = st.chat_input("Ask about your knowledge base…", max_chars=4000)
if prompt:
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        result = None
        try:
            if streaming:
                status = st.status("Analyzing query...", expanded=True)
                answer_box = st.empty()
                tokens: list[str] = []
                for event in client.chat_stream(prompt, conversation_id):
                    name, data = event["event"], event["data"]
                    if name == "token":
                        tokens.append(data.get("text", ""))
                        answer_box.markdown("".join(tokens) + "▌")
                    elif name == "completed":
                        result = data
                    elif name == "error":
                        err = data.get("error", {})
                        st.error(f"{err.get('code')}: {err.get('message')}")
                    else:
                        line = describe_event(name, data)
                        if line:
                            status.write(line)
                            if name == "retrieval_started":
                                status.update(label=f"{data.get('strategy', '').title()} search…")
                            elif name == "reasoning":
                                status.update(label=data.get("message", "Reasoning…"))
                if result:
                    status.update(label=f"Done · {result['retrieval_strategy']} · confidence {result['confidence']:.2f}",
                                  state="complete", expanded=False)
                    answer_box.markdown(result["answer"])
                else:
                    status.update(label="Failed", state="error")
            else:
                with st.spinner("Agent is working… (analyze → retrieve → grade → generate → verify)"):
                    result = client.chat(prompt, conversation_id)
                st.markdown(result["answer"])
        except APIError as exc:
            handle_api_error(exc)
        if result:
            render_assistant_result(result, show_answer=False)
            if result.get("cached"):
                st.caption("⚡ Served from the answer cache")
            render_feedback(result.get("message_id"))
            st.session_state["conversation_id"] = result["conversation_id"]
            st.session_state.pop("export", None)
            messages.append({"role": "assistant", "result": result})
    st.session_state["messages"] = messages
