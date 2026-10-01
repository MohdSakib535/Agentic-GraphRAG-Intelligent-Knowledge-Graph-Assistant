"""Neo4j repositories.

``GraphWriter`` (sync) is used by the Celery ingestion pipeline;
``GraphReader`` (async) is used by retrieval, the agent and the graph APIs.
Both require a tenant id on every call - there is no un-scoped access path.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from typing import Any

from neo4j import AsyncDriver, Driver, Query, RoutingControl
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired
from neo4j.graph import Node, Path, Relationship

from app.core.config import Settings
from app.core.errors import Neo4jUnavailable, RetrievalError, ValidationFailed
from app.core.logging import get_logger
from app.graph import queries as Q
from app.graph.schema import (
    CHUNK_FULLTEXT_INDEX,
    VECTOR_INDEX_NAME,
    safe_entity_label,
    safe_rel_type,
)

logger = get_logger(__name__)

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/&|])')


def _require_tenant(tenant_id: str) -> str:
    try:
        return str(uuid.UUID(str(tenant_id)))
    except (ValueError, TypeError) as exc:
        raise ValidationFailed("A valid tenant id is required") from exc


def lucene_query(terms: list[str]) -> str | None:
    escaped = [_LUCENE_SPECIAL.sub(r"\\\1", t) for t in terms if t]
    escaped = [t for t in escaped if t.strip()]
    return " OR ".join(escaped[:30]) if escaped else None


def filter_params(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = filters or {}
    return {
        "document_ids": filters.get("document_ids") or None,
        "filenames": filters.get("filenames") or None,
        "page_from": filters.get("page_from"),
        "page_to": filters.get("page_to"),
        "section": filters.get("section") or None,
    }


class GraphWriter:
    def __init__(self, driver: Driver, settings: Settings) -> None:
        self.driver = driver
        self.db = settings.neo4j_database
        self.batch_size = 500

    def _write(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        try:
            records, _, _ = self.driver.execute_query(cypher, params, database_=self.db)
        except (ServiceUnavailable, SessionExpired) as exc:
            raise Neo4jUnavailable() from exc
        return [r.data() for r in records]

    def upsert_document(self, tenant_id: str, document_id: str, filename: str, title: str | None, file_type: str) -> None:
        self._write(
            Q.UPSERT_DOCUMENT,
            tenant_id=_require_tenant(tenant_id),
            document_id=document_id,
            filename=filename,
            title=title,
            file_type=file_type,
        )

    def delete_document_chunks(self, tenant_id: str, document_id: str) -> None:
        self._write(Q.DELETE_DOCUMENT_CHUNKS, tenant_id=_require_tenant(tenant_id), document_id=document_id)

    def write_chunks(self, tenant_id: str, document_id: str, chunks: list[dict[str, Any]]) -> int:
        tenant_id = _require_tenant(tenant_id)
        written = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            if any(c.get("tenant_id") != tenant_id for c in batch):
                raise ValidationFailed("Chunk tenant mismatch")
            rows = self._write(Q.WRITE_CHUNKS, tenant_id=tenant_id, document_id=document_id, chunks=batch)
            written += rows[0]["written"] if rows else 0
        return written

    def set_chunk_embeddings(self, tenant_id: str, rows: list[dict[str, Any]]) -> int:
        tenant_id = _require_tenant(tenant_id)
        written = 0
        for i in range(0, len(rows), self.batch_size):
            out = self._write(Q.SET_CHUNK_EMBEDDINGS, tenant_id=tenant_id, rows=rows[i : i + self.batch_size])
            written += out[0]["written"] if out else 0
        return written

    def upsert_entities(self, tenant_id: str, entities: list[dict[str, Any]]) -> int:
        tenant_id = _require_tenant(tenant_id)
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ent in entities:
            by_type[safe_entity_label(ent["type"])].append(ent)
        written = 0
        for label, rows in by_type.items():
            cypher = Q.UPSERT_ENTITIES.replace("{label}", label)  # label comes from the whitelist
            for i in range(0, len(rows), self.batch_size):
                out = self._write(cypher, tenant_id=tenant_id, entities=rows[i : i + self.batch_size])
                written += out[0]["written"] if out else 0
        return written

    def write_mentions(self, tenant_id: str, mentions: list[tuple[str, str]]) -> int:
        tenant_id = _require_tenant(tenant_id)
        rows = [{"chunk_id": c, "entity_id": e} for c, e in mentions]
        written = 0
        for i in range(0, len(rows), self.batch_size):
            out = self._write(Q.WRITE_MENTIONS, tenant_id=tenant_id, mentions=rows[i : i + self.batch_size])
            written += out[0]["written"] if out else 0
        return written

    def upsert_relationships(self, tenant_id: str, document_id: str, rels: list[dict[str, Any]]) -> int:
        tenant_id = _require_tenant(tenant_id)
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rel in rels:
            by_type[safe_rel_type(rel["type"])].append(rel)
        written = 0
        for rel_type, rows in by_type.items():
            cypher = Q.UPSERT_RELATIONSHIPS.replace("{rel_type}", rel_type)  # whitelisted
            for i in range(0, len(rows), self.batch_size):
                out = self._write(
                    cypher, tenant_id=tenant_id, document_id=document_id, rels=rows[i : i + self.batch_size]
                )
                written += out[0]["written"] if out else 0
        return written

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        tenant_id = _require_tenant(tenant_id)
        chunk_prefix = f"chk_{uuid.UUID(document_id).hex}_"
        self._write(Q.DELETE_DOCUMENT, tenant_id=tenant_id, document_id=document_id)
        self._write(Q.PRUNE_DOCUMENT_RELATIONSHIPS, tenant_id=tenant_id, document_id=document_id, chunk_prefix=chunk_prefix)
        self._write(Q.PRUNE_DOCUMENT_ENTITIES, tenant_id=tenant_id, document_id=document_id)

    # EntityLookup protocol (used by the resolver)
    def find_by_keys(self, tenant_id: str, keys: list[str]) -> list[dict[str, Any]]:
        if not keys:
            return []
        return self._write(Q.FIND_ENTITIES_BY_KEYS, tenant_id=_require_tenant(tenant_id), keys=keys)

    def find_candidates(self, tenant_id: str, entity_type: str, tokens: list[str], limit: int) -> list[dict[str, Any]]:
        if not tokens:
            return []
        return self._write(
            Q.FIND_ENTITY_CANDIDATES,
            tenant_id=_require_tenant(tenant_id),
            type=safe_entity_label(entity_type),
            tokens=tokens,
            limit=limit,
        )


def _clean_value(value: Any) -> Any:
    """Convert neo4j graph types into JSON-friendly dicts, dropping embeddings."""
    if isinstance(value, Node):
        props = {k: v for k, v in dict(value).items() if k not in {"embedding"}}
        return {"labels": sorted(value.labels), **props}
    if isinstance(value, Relationship):
        return {"type": value.type, **{k: v for k, v in dict(value).items()}}
    if isinstance(value, Path):
        return {"nodes": [_clean_value(n) for n in value.nodes], "relationships": [_clean_value(r) for r in value.relationships]}
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_value(v) for k, v in value.items()}
    if hasattr(value, "iso_format"):
        return value.iso_format()
    return value


def _foreign_tenant(value: Any, tenant_id: str) -> bool:
    """True if a returned value contains a node/relationship belonging to another tenant."""
    if isinstance(value, (Node, Relationship)):
        owner = value.get("tenant_id")
        return owner is not None and owner != tenant_id
    if isinstance(value, Path):
        return any(_foreign_tenant(n, tenant_id) for n in value.nodes)
    if isinstance(value, (list, tuple)):
        return any(_foreign_tenant(v, tenant_id) for v in value)
    if isinstance(value, dict):
        return any(_foreign_tenant(v, tenant_id) for v in value.values())
    return False


class GraphReader:
    def __init__(self, driver: AsyncDriver, settings: Settings) -> None:
        self.driver = driver
        self.settings = settings
        self.db = settings.neo4j_database
        self.timeout = settings.neo4j_query_timeout_seconds

    async def _read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        try:
            records, _, _ = await self.driver.execute_query(
                Query(cypher, timeout=self.timeout),  # type: ignore[arg-type]
                params,
                database_=self.db,
                routing_=RoutingControl.READ,
            )
        except (ServiceUnavailable, SessionExpired) as exc:
            raise Neo4jUnavailable() from exc
        except Neo4jError as exc:
            logger.warning("neo4j_read_failed", extra={"code": exc.code})
            raise RetrievalError() from exc
        return [r.data() for r in records]

    # ------------------------------------------------------------- chunks
    async def count_chunks(self, tenant_id: str) -> int:
        rows = await self._read(Q.COUNT_TENANT_CHUNKS, tenant_id=_require_tenant(tenant_id))
        return int(rows[0]["n"]) if rows else 0

    async def vector_search(
        self, tenant_id: str, embedding: list[float], top_k: int, filters: dict[str, Any] | None = None,
        exact_scan_max: int = 20000,
    ) -> list[dict[str, Any]]:
        tenant_id = _require_tenant(tenant_id)
        params = {"tenant_id": tenant_id, "embedding": embedding, "top_k": top_k, **filter_params(filters)}
        if await self.count_chunks(tenant_id) <= exact_scan_max:
            return await self._read(Q.VECTOR_SEARCH_EXACT, **params)
        # Large tenant: ANN with over-fetch; widen until we have top_k tenant hits or hit the cap.
        candidates = top_k * self.settings.vector_oversample_factor
        rows: list[dict[str, Any]] = []
        while True:
            rows = await self._read(Q.VECTOR_SEARCH_ANN, index_name=VECTOR_INDEX_NAME, candidates=candidates, **params)
            if len(rows) >= top_k or candidates >= 5000:
                return rows
            candidates *= 4

    async def fulltext_chunks(
        self, tenant_id: str, terms: list[str], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        query = lucene_query(terms)
        if not query:
            return []
        return await self._read(
            Q.FULLTEXT_CHUNKS,
            tenant_id=_require_tenant(tenant_id),
            index_name=CHUNK_FULLTEXT_INDEX,
            query=query,
            candidates=top_k * self.settings.vector_oversample_factor,
            top_k=top_k,
            **filter_params(filters),
        )

    async def chunks_by_ids(self, tenant_id: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        return await self._read(Q.CHUNKS_BY_IDS, tenant_id=_require_tenant(tenant_id), chunk_ids=chunk_ids)

    async def chunks_mentioning(self, tenant_id: str, entity_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        return await self._read(
            Q.CHUNKS_MENTIONING,
            tenant_id=_require_tenant(tenant_id),
            entity_ids=entity_ids,
            n_entities=float(len(entity_ids)),
            top_k=top_k,
        )

    # ----------------------------------------------------------- entities
    async def link_entities_exact(self, tenant_id: str, names: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not names:
            return []
        return await self._read(Q.LINK_ENTITIES_EXACT, tenant_id=_require_tenant(tenant_id), names=names)

    async def link_entities_fuzzy(self, tenant_id: str, names: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
        names = [n for n in names if n.get("tokens")]
        if not names:
            return []
        return await self._read(Q.LINK_ENTITIES_FUZZY, tenant_id=_require_tenant(tenant_id), names=names, limit=limit)

    async def entities_in_text(self, tenant_id: str, normalized_text: str) -> list[dict[str, Any]]:
        return await self._read(Q.ENTITIES_IN_TEXT, tenant_id=_require_tenant(tenant_id), text=normalized_text)

    async def neighborhood(
        self, tenant_id: str, entity_ids: list[str], rel_types: list[str] | None, limit: int
    ) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        rel_types = [safe_rel_type(r) for r in rel_types] if rel_types else None
        return await self._read(
            Q.NEIGHBORHOOD, tenant_id=_require_tenant(tenant_id), entity_ids=entity_ids, rel_types=rel_types, limit=limit
        )

    async def common_neighbors(
        self, tenant_id: str, anchors: list[dict[str, Any]], min_anchors: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Entities adjacent to >= ``min_anchors`` anchors; each anchor ``{id, rels}`` may restrict edge types."""
        if len(anchors) < 2:
            return []
        clean = [{"id": a["id"], "rels": [safe_rel_type(r) for r in a["rels"]] if a.get("rels") else None} for a in anchors]
        return await self._read(
            Q.COMMON_NEIGHBORS, tenant_id=_require_tenant(tenant_id), anchors=clean,
            entity_ids=[a["id"] for a in clean], min_anchors=min_anchors, limit=limit,
        )

    async def paths_between(self, tenant_id: str, entity_ids: list[str], max_hops: int, limit: int = 25) -> list[dict[str, Any]]:
        if len(entity_ids) < 2:
            return []
        hops = max(1, min(int(max_hops), 4))
        cypher = Q.PATHS_BETWEEN.replace("{max_hops}", str(hops))
        return await self._read(cypher, tenant_id=_require_tenant(tenant_id), entity_ids=entity_ids, limit=limit)

    # ------------------------------------------------------------ explorer
    async def stats(self, tenant_id: str) -> dict[str, Any]:
        tenant_id = _require_tenant(tenant_id)
        rows = await self._read(Q.GRAPH_STATS, tenant_id=tenant_id)
        by_type = await self._read(Q.ENTITIES_BY_TYPE, tenant_id=tenant_id)
        rel_types = await self._read(Q.RELATIONSHIPS_BY_TYPE, tenant_id=tenant_id)
        base = rows[0] if rows else {"entities": 0, "relationships": 0, "chunks": 0, "documents": 0}
        return {
            **base,
            "entities_by_type": {r["type"]: r["n"] for r in by_type},
            "relationships_by_type": {r["type"]: r["n"] for r in rel_types if r["type"] not in {"MENTIONS", "CONTAINS"}},
        }

    async def search_entities(self, tenant_id: str, q: str | None, types: list[str] | None, limit: int, offset: int) -> list[dict[str, Any]]:
        types = [safe_entity_label(t) for t in types] if types else None
        return await self._read(
            Q.SEARCH_ENTITIES, tenant_id=_require_tenant(tenant_id), q=q or None, types=types, limit=limit, offset=offset
        )

    async def entity(self, tenant_id: str, entity_id: str) -> dict[str, Any] | None:
        rows = await self._read(Q.ENTITY_BY_ID, tenant_id=_require_tenant(tenant_id), entity_id=entity_id)
        return rows[0] if rows else None

    async def entity_sources(self, tenant_id: str, entity_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self._read(Q.ENTITY_SOURCES, tenant_id=_require_tenant(tenant_id), entity_id=entity_id, limit=limit)

    async def subgraph(self, tenant_id: str, entity_id: str | None, types: list[str] | None, limit: int, depth: int = 1) -> list[dict[str, Any]]:
        tenant_id = _require_tenant(tenant_id)
        types = [safe_entity_label(t) for t in types] if types else None
        if entity_id and depth > 1:
            cypher = Q.EXPAND_SUBGRAPH.replace("{depth}", str(max(1, min(depth, 3))))
            return await self._read(cypher, tenant_id=tenant_id, entity_id=entity_id, limit=limit)
        return await self._read(Q.SUBGRAPH, tenant_id=tenant_id, entity_id=entity_id, types=types, limit=limit)

    # ------------------------------------------------- validated Text2Cypher
    async def run_validated_readonly(self, cypher: str, params: dict[str, Any], tenant_id: str, row_limit: int) -> list[dict[str, Any]]:
        """Execute an already-validated, tenant-rewritten query in a READ transaction.

        Defense in depth: (1) the validator rejected write clauses and procedures,
        (2) the query runs in a read-only transaction so the server refuses writes,
        (3) rows referencing nodes of another tenant are dropped.
        """
        tenant_id = _require_tenant(tenant_id)

        async def work(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(cypher, {**params, "tenant_id": tenant_id})
            rows: list[dict[str, Any]] = []
            async for record in result:
                values = dict(record.items())
                if _foreign_tenant(values, tenant_id):
                    logger.warning("text2cypher_foreign_tenant_row_dropped")
                    continue
                rows.append(_clean_value(values))
                if len(rows) >= row_limit:
                    break
            return rows

        from neo4j import unit_of_work

        @unit_of_work(timeout=self.timeout)
        async def bounded(tx: Any) -> list[dict[str, Any]]:
            return await work(tx)

        try:
            async with self.driver.session(database=self.db, default_access_mode="READ") as session:
                return await session.execute_read(bounded)
        except (ServiceUnavailable, SessionExpired) as exc:
            raise Neo4jUnavailable() from exc
        except Neo4jError as exc:
            logger.warning("text2cypher_execution_failed", extra={"code": exc.code})
            raise RetrievalError("Generated graph query failed to execute") from exc
