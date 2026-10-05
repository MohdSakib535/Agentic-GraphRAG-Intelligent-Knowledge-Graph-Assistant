"""Re-embed every chunk with the configured embedding provider (after switching OpenAI <-> Bedrock).

    python -m app.scripts.reindex_embeddings           # inside the backend container: make reindex

Drops and recreates the Neo4j vector index with the current EMBEDDING_DIMENSIONS, then re-embeds chunk text in
batches, tenant by tenant. Graph entities, relationships and chunks are untouched - no re-extraction or LLM calls.
Answer and retrieval caches are invalidated afterwards. Safe to re-run.
"""

from __future__ import annotations

import sys

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.neo4j import ensure_schema_sync, get_sync_driver, vector_index_dimensions_sync
from app.db.redis import invalidate_tenant_cache_sync
from app.graph.repository import GraphWriter
from app.graph.schema import VECTOR_INDEX_NAME
from app.ingestion.embedding import build_embedder

logger = get_logger("reindex")

TENANTS = "MATCH (c:Chunk) RETURN DISTINCT c.tenant_id AS tenant_id"
PAGE = ("MATCH (c:Chunk {tenant_id: $tenant_id}) WHERE c.id > $after "
        "RETURN c.id AS id, c.text AS text ORDER BY c.id LIMIT $limit")


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    driver = get_sync_driver(settings)
    db = settings.neo4j_database
    embedder = build_embedder(settings)
    before = vector_index_dimensions_sync(driver, settings)
    driver.execute_query(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS", database_=db)
    ensure_schema_sync(driver, settings)
    writer = GraphWriter(driver, settings)
    tenants = [r["tenant_id"] for r in driver.execute_query(TENANTS, database_=db)[0]]
    total = 0
    for tenant_id in tenants:
        after = ""
        while True:
            rows = driver.execute_query(PAGE, {"tenant_id": tenant_id, "after": after,
                                               "limit": settings.embedding_batch_size * 4}, database_=db)[0]
            if not rows:
                break
            vectors = embedder.embed_documents([r["text"] or "" for r in rows])
            total += writer.set_chunk_embeddings(tenant_id, [{"id": r["id"], "embedding": v}
                                                             for r, v in zip(rows, vectors, strict=True)])
            after = rows[-1]["id"]
        invalidate_tenant_cache_sync(settings.redis_url, tenant_id)
    print(f"Re-embedded {total} chunks in {len(tenants)} tenant(s) with {embedder.name} "
          f"({settings.embedding_dimensions} dims; index was {before}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
