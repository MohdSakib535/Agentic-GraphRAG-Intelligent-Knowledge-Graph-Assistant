"""Builds the tenant knowledge graph from resolved extraction output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.graph.repository import GraphWriter
from app.ingestion.entity_resolver import ResolutionResult


@dataclass
class GraphBuildStats:
    chunks: int
    entities: int
    relationships: int
    mentions: int


class GraphBuilder:
    def __init__(self, writer: GraphWriter) -> None:
        self.writer = writer

    def reset_document(self, tenant_id: str, document_id: str) -> None:
        """Remove a previous ingestion of this document (idempotent retries / re-processing)."""
        self.writer.delete_document(tenant_id, document_id, keep_curated=True)

    def build(
        self,
        *,
        tenant_id: str,
        document_id: str,
        filename: str,
        title: str | None,
        file_type: str,
        chunks: list[dict[str, Any]],
        resolution: ResolutionResult,
    ) -> GraphBuildStats:
        """Create Document, Chunk, Entity nodes plus CONTAINS / MENTIONS / semantic relationships."""
        self.writer.upsert_document(tenant_id, document_id, filename, title, file_type)
        n_chunks = self.writer.write_chunks(tenant_id, document_id, chunks)
        entity_rows = [e.as_record(tenant_id, document_id) for e in resolution.entities]
        n_entities = self.writer.upsert_entities(tenant_id, entity_rows)
        n_mentions = self.writer.write_mentions(tenant_id, resolution.mentions)
        rel_rows = [
            {
                "source_id": r.source_id,
                "target_id": r.target_id,
                "type": r.type,
                "evidence": r.evidence[:600],
                "chunk_ids": sorted(r.chunk_ids),
            }
            for r in resolution.relationships
        ]
        n_rels = self.writer.upsert_relationships(tenant_id, document_id, rel_rows)
        self.writer.prune_orphans(tenant_id)
        return GraphBuildStats(n_chunks, n_entities, n_rels, n_mentions)

    def index_embeddings(self, tenant_id: str, chunk_ids: list[str], embeddings: list[list[float]]) -> int:
        """Attach embeddings to chunk nodes; the HNSW vector index picks them up."""
        rows = [{"id": cid, "embedding": emb} for cid, emb in zip(chunk_ids, embeddings, strict=True)]
        return self.writer.set_chunk_embeddings(tenant_id, rows)
