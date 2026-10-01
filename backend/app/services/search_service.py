"""Direct (non-agentic) search and graph exploration for the API and UI."""

from __future__ import annotations

from typing import Any

from app.core.container import Container
from app.core.errors import NotFoundError
from app.schemas.search import EntityDetail, EntityOut, GraphFact, GraphStats, SearchRequest, SearchResponse, SubgraphOut


def _fact(row: dict[str, Any]) -> GraphFact:
    return GraphFact(
        source_id=row.get("source_id"), target_id=row.get("target_id"),
        source=row["source"], source_type=row["source_type"], relationship=row["relationship"], target=row["target"],
        target_type=row["target_type"], evidence=row.get("evidence"), chunk_ids=list(row.get("chunk_ids") or []),
        document_ids=list(row.get("document_ids") or []),
    )


class SearchService:
    def __init__(self, container: Container) -> None:
        self.container = container
        self.reader = container.reader

    async def search(self, tenant_id: str, request: SearchRequest) -> SearchResponse:
        from app.retrieval.query_parsing import expected_answer_type, relation_hints

        filters = request.filters.model_dump(exclude_none=True) if request.filters else None
        result = await self.container.retrieval.retrieve(
            request.strategy, request.query, tenant_id, top_k=request.top_k, filters=filters, rerank=request.rerank,
            relations=relation_hints(request.query), answer_type=expected_answer_type(request.query),
        )
        return SearchResponse(
            query=request.query, strategy=result.strategy, chunks=result.chunks, facts=result.facts,
            linked_entities=[e.model_dump() for e in result.linked_entities], cypher=result.cypher,
            latency_ms=result.latency_ms, cached=result.cached,
        )

    async def stats(self, tenant_id: str) -> GraphStats:
        return GraphStats(**await self.reader.stats(tenant_id))

    async def entities(self, tenant_id: str, q: str | None, types: list[str] | None, limit: int, offset: int) -> list[EntityOut]:
        rows = await self.reader.search_entities(tenant_id, q, types, limit, offset)
        return [EntityOut(**r) for r in rows]

    async def entity_detail(self, tenant_id: str, entity_id: str) -> EntityDetail:
        row = await self.reader.entity(tenant_id, entity_id)
        if row is None:
            raise NotFoundError("Entity not found", code="ENTITY_NOT_FOUND")
        rels = await self.reader.subgraph(tenant_id, entity_id, None, 200)
        neighbors: dict[str, EntityOut] = {}
        for r in rels:
            for nid, name, etype in ((r["source_id"], r["source"], r["source_type"]), (r["target_id"], r["target"], r["target_type"])):
                if nid != entity_id:
                    neighbors.setdefault(nid, EntityOut(id=nid, name=name, type=etype))
        sources = await self.reader.entity_sources(tenant_id, entity_id)
        return EntityDetail(entity=EntityOut(**row), relationships=[_fact(r) for r in rels],
                            neighbors=list(neighbors.values()), sources=sources)

    async def subgraph(self, tenant_id: str, entity_id: str | None, types: list[str] | None, limit: int, depth: int) -> SubgraphOut:
        rows = await self.reader.subgraph(tenant_id, entity_id, types, limit, depth)
        nodes: dict[str, EntityOut] = {}
        for r in rows:
            nodes.setdefault(r["source_id"], EntityOut(id=r["source_id"], name=r["source"], type=r["source_type"]))
            nodes.setdefault(r["target_id"], EntityOut(id=r["target_id"], name=r["target"], type=r["target_type"]))
        for node in nodes.values():
            node.degree = sum(1 for r in rows if node.id in (r["source_id"], r["target_id"]))
        return SubgraphOut(nodes=list(nodes.values()), edges=[_fact(r) for r in rows])
