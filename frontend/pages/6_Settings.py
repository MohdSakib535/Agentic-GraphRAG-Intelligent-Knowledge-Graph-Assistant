"""Settings: effective, non-secret configuration and service health."""

from __future__ import annotations

import streamlit as st

from services.api_client import APIError
from utils.session import handle_api_error, require_auth, sidebar_user_box

st.set_page_config(page_title="Settings · Agentic GraphRAG", page_icon="⚙️", layout="wide")
client = require_auth()
sidebar_user_box()
st.title("⚙️ Settings")
st.caption("Configuration is managed through environment variables on the server. Secrets are never exposed here.")

try:
    cfg = client.settings()
    health = client.health()
except APIError as exc:
    handle_api_error(exc)
    st.stop()

a, b, c = st.columns(3)
with a:
    st.subheader("Models")
    st.metric("LLM provider", cfg["llm_provider"])
    st.metric("LLM model", cfg["llm_model"])
    st.metric("Embedding model", cfg["embedding_model"])
    st.metric("Embedding dimensions", cfg["embedding_dimensions"])
with b:
    st.subheader("Retrieval")
    st.metric("Top K", cfg["top_k"])
    st.metric("Chunk size (tokens)", cfg["chunk_size"])
    st.metric("Chunk overlap (tokens)", cfg["chunk_overlap"])
    st.metric("Retrieval threshold", cfg["retrieval_threshold"])
    st.metric("Reranker", cfg["reranker"])
with c:
    st.subheader("Agent & limits")
    st.metric("Max agent retries", cfg["agent_max_retries"])
    st.metric("Max graph hops", cfg["graph_max_hops"])
    st.metric("Text2Cypher", "enabled" if cfg["text2cypher_enabled"] else "disabled")
    st.metric("Max upload (MB)", cfg["max_upload_size_mb"])
    st.write("Rate limits:", cfg["rate_limits"])

st.subheader("Features")
obs = cfg.get("observability") or {}
features = {
    "OCR for scanned PDFs": cfg.get("ocr_enabled"),
    "Answer cache (Redis)": cfg.get("answer_cache_enabled"),
    "Sign in with Google": cfg.get("google_login_enabled"),
    "OpenTelemetry tracing": obs.get("tracing"),
    "Prometheus metrics (/metrics)": obs.get("metrics"),
    "LangSmith tracing": obs.get("langsmith"),
}
fc = st.columns(3)
for i, (label, enabled) in enumerate(features.items()):
    fc[i % 3].markdown(f"{'🟢' if enabled else '⚪'} {label}")

st.subheader("Service health")
hc = st.columns(len(health["checks"]))
for col, (name, check) in zip(hc, health["checks"].items(), strict=True):
    col.metric(name.title(), "🟢 ok" if check["status"] == "ok" else "🔴 down")
