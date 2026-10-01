"""Vector (dense) and keyword (full-text) chunk retrieval - always tenant-scoped."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.repository import GraphReader
from app.ingestion.embedding import Embedder
from app.schemas.search import ChunkHit
from app.utils.text import content_terms

logger = get_logger(__name__)


def row_to_hit(row: dict[str, Any], retriever: str, score: float | None = None) -> ChunkHit:
    return ChunkHit(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        text=row["text"] or "",
        score=float(row.get("score") or 0.0) if score is None else score,
        source_filename=row.get("source_filename"),
        page_number=row.get("page_number"),
        section=row.get("section"),
        retrievers=[retriever],
        metadata={
            "chunk_index": row.get("chunk_index"),
            "document_title": row.get("document_title"),
            f"{retriever}_score": float(row.get("score") or 0.0),
        },
    )


class VectorRetriever:
    def __init__(self, reader: GraphReader, embedder: Embedder, settings: Settings) -> None:
        self.reader = reader
        self.embedder = embedder
        self.settings = settings

    async def similarity_search(
        self, query: str, tenant_id: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[ChunkHit]:
        embedding = await self.embedder.aembed_query(query)
        rows = await self.reader.vector_search(tenant_id, embedding, top_k, filters)
        return [row_to_hit(r, "vector") for r in rows]

    async def keyword_search(
        self, query: str, tenant_id: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[ChunkHit]:
        terms = list(dict.fromkeys(content_terms(query)))
        rows = await self.reader.fulltext_chunks(tenant_id, terms, top_k, filters)
        if not rows:
            return []
        top = max(float(r["score"] or 0) for r in rows) or 1.0
        return [row_to_hit(r, "keyword", float(r["score"] or 0) / top) for r in rows]
