"""Vector (dense) and keyword (full-text) chunk retrieval - always tenant-scoped."""

from __future__ import annotations

import base64
from array import array
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.metrics import record_cache
from app.db.redis import TenantCache, query_hash
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


def _pack(vector: list[float]) -> str:
    return base64.b64encode(array("f", vector).tobytes()).decode()


def _unpack(raw: str) -> list[float]:
    values = array("f")
    values.frombytes(base64.b64decode(raw))
    return values.tolist()


class VectorRetriever:
    def __init__(self, reader: GraphReader, embedder: Embedder, settings: Settings, cache: TenantCache | None = None) -> None:
        self.reader = reader
        self.embedder = embedder
        self.settings = settings
        self.cache = cache
        self.embedding_cache_hits = 0

    async def embed_query(self, query: str, tenant_id: str) -> list[float]:
        """Query embedding with a tenant-scoped Redis cache (float32, base64) - saves an embedding API call."""
        if self.cache is None:
            return await self.embedder.aembed_query(query)
        digest = query_hash(self.embedder.name, self.embedder.dimensions, " ".join(query.split()))
        cached = await self.cache.get_static(tenant_id, "embedding", digest)
        if cached:
            vector = _unpack(cached)
            if len(vector) == self.embedder.dimensions:
                self.embedding_cache_hits += 1
                record_cache("embedding", hit=True)
                return vector
        record_cache("embedding", hit=False)
        vector = await self.embedder.aembed_query(query)
        await self.cache.set_static(tenant_id, "embedding", digest, _pack(vector), self.settings.embedding_cache_ttl_seconds)
        return vector

    async def similarity_search(
        self, query: str, tenant_id: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[ChunkHit]:
        embedding = await self.embed_query(query, tenant_id)
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
