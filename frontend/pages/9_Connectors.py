"""Content connectors (admin): keep a Google Drive folder in sync with the knowledge base."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.api_client import APIError
from utils.session import handle_api_error, parse_groups, require_admin, sidebar_user_box

st.set_page_config(page_title="Connectors · Agentic GraphRAG", page_icon="🔌", layout="wide")
client = require_admin()
sidebar_user_box()

st.title("🔌 Connectors")
st.caption("Google Drive: share a folder with a Google Cloud service account (Viewer access) and paste the "
           "service-account JSON key below. PDFs, Word, text, Markdown files and Google Docs in the folder and its "
           "subfolders are ingested. Syncs are incremental: changed files are re-processed and files removed from "
           "Drive are removed from the knowledge base. The key is stored encrypted and never shown again.")

try:
    connectors = client.connectors()
except APIError as exc:
    handle_api_error(exc)
    connectors = []

STATUS = {"IDLE": "✅ idle", "RUNNING": "🔄 syncing", "QUEUED": "⏳ queued", "FAILED": "❌ failed"}
for c in connectors:
    stats = c.get("last_sync_stats") or {}
    with st.container(border=True):
        top = st.columns([3, 1, 1, 1])
        top[0].markdown(f"**{c['name']}** · Google Drive folder `{c['config'].get('folder_name') or c['config'].get('folder_id')}`")
        top[0].caption(f"Service account: {c['config'].get('service_account')} · "
                       f"Groups: {', '.join(c['access_groups']) or 'whole workspace'}")
        top[1].markdown(STATUS.get(c["status"], c["status"]))
        if top[2].button("Sync now", key=f"sync-{c['id']}"):
            try:
                client.sync_connector(c["id"])
            except APIError as exc:
                handle_api_error(exc)
            else:
                st.toast("Sync started.")
                st.rerun()
        with top[3].popover("Remove"):
            purge = st.checkbox("Also delete its documents", key=f"purge-{c['id']}")
            if st.button("Confirm removal", key=f"del-{c['id']}", type="primary"):
                try:
                    client.delete_connector(c["id"], purge)
                except APIError as exc:
                    handle_api_error(exc)
                else:
                    st.rerun()
        if c.get("last_sync_at"):
            st.caption(f"Last sync {c['last_sync_at'][:19].replace('T', ' ')} UTC")
            st.dataframe(pd.DataFrame([{k: stats.get(k, 0) for k in ("created", "updated", "unchanged", "removed")}
                                       | {"skipped": len(stats.get("skipped") or []), "failed": len(stats.get("failed") or [])}]),
                         hide_index=True)
        if c.get("last_error"):
            st.error(c["last_error"])

st.subheader("Connect a Google Drive folder")
with st.form("new-connector", clear_on_submit=True):
    name = st.text_input("Name", placeholder="Engineering wiki")
    folder = st.text_input("Folder ID", help="The last part of the folder URL: drive.google.com/drive/folders/<ID>")
    key = st.text_area("Service-account JSON key", height=160, placeholder='{"type": "service_account", ...}')
    groups = st.text_input("Restrict synced documents to groups (optional)")
    sync_now = st.checkbox("Start the first sync now", value=True)
    if st.form_submit_button("Connect", type="primary"):
        folder_id = folder.strip().rstrip("/").split("/")[-1].split("?")[0]
        try:
            client.create_connector(name.strip() or "Google Drive", folder_id, key.strip(), parse_groups(groups), sync_now)
        except APIError as exc:
            handle_api_error(exc)
        else:
            st.success("Connected.")
            st.rerun()
