"""Document- and chunk-level metadata."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.ingestion.chunker import Chunk
from app.ingestion.parser import ParsedDocument
from app.utils.ids import chunk_id
from app.utils.text import content_terms
from app.utils.tokens import count_tokens, tokenizer_name


def document_metadata(parsed: ParsedDocument, filename: str, file_type: str) -> dict[str, Any]:
    text = parsed.text
    terms = [t for t in content_terms(text) if len(t) > 3 and not t.isdigit()]
    sections = []
    for block in parsed.blocks:
        if block.section and block.section not in sections:
            sections.append(block.section)
    return {
        "filename": filename,
        "file_type": file_type,
        "title": parsed.title,
        "page_count": parsed.page_count,
        "sections": sections[:100],
        "word_count": len(text.split()),
        "token_count": count_tokens(text),
        "tokenizer": tokenizer_name(),
        "keywords": [t for t, _ in Counter(terms).most_common(15)],
        **{k: v for k, v in parsed.metadata.items() if isinstance(v, (str, int, float, bool))},
    }


def chunk_records(
    chunks: list[Chunk], *, tenant_id: str, document_id: str, source_filename: str, title: str | None
) -> list[dict[str, Any]]:
    """Flatten chunks into the property maps stored on ``(:Chunk)`` nodes."""
    return [
        {
            "id": chunk_id(document_id, c.index),
            "tenant_id": tenant_id,
            "document_id": document_id,
            "chunk_index": c.index,
            "text": c.text,
            "token_count": c.token_count,
            "page_number": c.page_number,
            "page_end": c.page_end,
            "section": c.section,
            "source_filename": source_filename,
            "document_title": title,
        }
        for c in chunks
    ]
