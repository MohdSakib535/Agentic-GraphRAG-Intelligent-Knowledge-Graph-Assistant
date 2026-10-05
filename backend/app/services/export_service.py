"""Export a conversation (questions, answers and citations) as Markdown or PDF."""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from typing import Any

from app.models.conversation import Conversation
from app.models.message import Message


def _source_line(src: dict[str, Any]) -> str:
    if src.get("kind") == "graph" and not src.get("source_filename"):
        loc = "knowledge graph"
    else:
        loc = src.get("source_filename") or "document"
        if src.get("page_number"):
            loc += f", page {src['page_number']}"
        if src.get("section"):
            loc += f", § {src['section']}"
    return f"[{src.get('index')}] {loc}"


def to_markdown(conversation: Conversation, messages: list[Message]) -> str:
    lines = [f"# {conversation.title}", "",
             f"_Exported {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · Agentic GraphRAG_", ""]
    for msg in messages:
        if msg.role == "user":
            lines += [f"## Q: {msg.content}", ""]
            continue
        lines += [msg.content, ""]
        meta = [f"strategy {msg.retrieval_strategy}"] if msg.retrieval_strategy else []
        if msg.confidence is not None:
            meta.append(f"confidence {msg.confidence:.2f}")
        if meta:
            lines += [f"_{' · '.join(meta)}_", ""]
        if msg.sources:
            lines.append("**Sources**")
            lines += [f"- {_source_line(s)}" + (f" — {s['snippet'][:200]}" if s.get("snippet") else "")
                      for s in msg.sources]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_CSS = """
body { font-family: sans-serif; font-size: 10pt; }
h1 { font-size: 16pt; margin-bottom: 4pt; }
h2 { font-size: 11pt; color: #3b2b8f; margin-top: 12pt; }
p.meta { color: #666666; font-size: 8pt; }
p.answer { white-space: pre-wrap; }
li { font-size: 8.5pt; color: #333333; }
"""


def to_html(conversation: Conversation, messages: list[Message]) -> str:
    parts = [f"<h1>{escape(conversation.title)}</h1>",
             f"<p class='meta'>Exported {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · Agentic GraphRAG</p>"]
    for msg in messages:
        if msg.role == "user":
            parts.append(f"<h2>Q: {escape(msg.content)}</h2>")
            continue
        parts.append("<p class='answer'>" + escape(msg.content).replace("\n", "<br/>") + "</p>")
        meta = []
        if msg.retrieval_strategy:
            meta.append(f"strategy {escape(msg.retrieval_strategy)}")
        if msg.confidence is not None:
            meta.append(f"confidence {msg.confidence:.2f}")
        if meta:
            parts.append(f"<p class='meta'>{' · '.join(meta)}</p>")
        if msg.sources:
            parts.append("<p><b>Sources</b></p><ul>")
            parts += [f"<li>{escape(_source_line(s))}</li>" for s in msg.sources]
            parts.append("</ul>")
    return "".join(parts)


def to_pdf(conversation: Conversation, messages: list[Message]) -> bytes:
    """Render with PyMuPDF's Story layout engine (HTML subset), paginating automatically."""
    import io

    import pymupdf

    story = pymupdf.Story(html=to_html(conversation, messages), user_css=_CSS)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (50, 50, -50, -50)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()
