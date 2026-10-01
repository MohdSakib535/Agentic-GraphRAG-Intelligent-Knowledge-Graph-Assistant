"""Citation rendering."""

from __future__ import annotations

from typing import Any

import streamlit as st


def source_label(src: dict[str, Any]) -> str:
    if src.get("kind") == "graph" and not src.get("source_filename"):
        return "knowledge graph"
    parts = [src.get("source_filename") or "document"]
    if src.get("page_number"):
        parts.append(f"page {src['page_number']}")
    if src.get("section"):
        parts.append(f"§ {src['section']}")
    return ", ".join(parts)


def render_citations(sources: list[dict[str, Any]], expanded: bool = False) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=expanded):
        for src in sources:
            st.markdown(f"**[{src.get('index')}]** {source_label(src)}")
            if src.get("snippet"):
                st.caption(src["snippet"])
