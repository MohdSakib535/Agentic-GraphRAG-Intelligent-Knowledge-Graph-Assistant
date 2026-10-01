"""Interactive knowledge-graph visualisation (pyvis / vis.js).

Entity names come from documents (and possibly LLM extraction), so tooltip
HTML is escaped (labels are drawn on a canvas, not parsed as HTML) and the graph is rendered from a ``data:`` URL iframe, which has an
opaque origin and cannot reach the Streamlit app. Kept independent of the pages
so another renderer can be swapped in.
"""

from __future__ import annotations

import base64
from html import escape
from typing import Any

import streamlit as st

TYPE_COLORS = {
    "Person": "#4C78A8",
    "Project": "#F58518",
    "Technology": "#54A24B",
    "Company": "#B279A2",
    "Department": "#E45756",
    "Product": "#72B7B2",
    "Location": "#9D755D",
    "Concept": "#BAB0AC",
}


def build_graph_html(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], highlight: str | None = None,
                     height: int = 620) -> str:
    from pyvis.network import Network

    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222",
                  cdn_resources="remote")
    net.barnes_hut(gravity=-4000, central_gravity=0.25, spring_length=140)
    for node in nodes:
        is_focus = highlight is not None and node["id"] == highlight
        net.add_node(
            node["id"], label=node["name"], title=escape(f"{node['type']}: {node['name']}"),
            color=TYPE_COLORS.get(node["type"], "#999999"), shape="dot",
            size=28 if is_focus else 12 + min(int(node.get("degree") or 0), 10) * 1.5,
            borderWidth=4 if is_focus else 1,
        )
    known = {n["id"] for n in nodes}
    for edge in edges:
        src, tgt = edge.get("source_id"), edge.get("target_id")
        if src in known and tgt in known:
            net.add_edge(src, tgt, label=edge["relationship"],
                         title=escape(edge.get("evidence") or edge["relationship"]),
                         arrows="to", font={"size": 10, "align": "middle"})
    return net.generate_html(notebook=False)


def render_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], highlight: str | None = None,
                 height: int = 620) -> None:
    html = build_graph_html(nodes, edges, highlight, height)
    st.iframe("data:text/html;base64," + base64.b64encode(html.encode()).decode(), height=height + 20)
