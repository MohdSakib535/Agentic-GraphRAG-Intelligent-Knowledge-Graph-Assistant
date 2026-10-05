"""Admin curation of the knowledge graph: fix, merge and delete entities and relationships.

Every edit runs in a single Neo4j write transaction and is tenant-scoped. Labels and
relationship types are only ever interpolated from the schema whitelists.

Edits survive re-ingestion:
- renamed / re-typed / merged-away entities leave a ``merged_keys`` entry, so the resolver maps
  the old surface form onto the surviving entity;
- relationships added by hand carry ``manual = true`` and are never pruned with a document.
Deleted entities and relationships are re-created if a document that states them is re-processed.
"""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver, AsyncManagedTransaction
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from app.core.config import Settings
from app.core.errors import ConflictError, Neo4jUnavailable, NotFoundError, ValidationFailed
from app.graph.repository import _require_tenant
from app.graph.schema import ENTITY_TYPES, RelationType, is_valid_relation, safe_entity_label, safe_rel_type
from app.ingestion.entity_resolver import canonical_key

_ALIASES_TEXT = "reduce(s = '', a IN coalesce(e.aliases, []) | s + ' ' + a)"
_UNION = "reduce(acc = coalesce({a}, []), x IN coalesce({b}, []) | CASE WHEN x IN acc THEN acc ELSE acc + x END)"

GET_ENTITY = """
MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
RETURN e.id AS id, e.name AS name, e.type AS type, e.normalized_name AS normalized_name,
       coalesce(e.description, '') AS description, coalesce(e.aliases, []) AS aliases,
       coalesce(e.merged_keys, []) AS merged_keys
"""

FIND_DUPLICATE = """
MATCH (e:Entity {tenant_id: $tenant_id, type: $type, normalized_name: $normalized_name})
WHERE e.id <> $entity_id RETURN e.id AS id, e.name AS name LIMIT 1
"""

UPDATE_ENTITY = f"""
MATCH (e:Entity {{id: $entity_id, tenant_id: $tenant_id}})
SET e.name = $name, e.normalized_name = $normalized_name, e.type = $type, e.description = $description,
    e.aliases = $aliases, e.merged_keys = $merged_keys, e.edited = true, e.updated_at = datetime()
SET e.aliases_text = {_ALIASES_TEXT}
REMOVE e:{':'.join(ENTITY_TYPES)}
SET e:{{label}}
RETURN e.id AS id
"""

MOVE_OUTGOING = f"""
MATCH (dup:Entity {{id: $dup_id, tenant_id: $tenant_id}})-[r:{{rel}}]->(x:Entity {{tenant_id: $tenant_id}})
MATCH (keep:Entity {{id: $keep_id, tenant_id: $tenant_id}})
WITH keep, dup, r, x WHERE x <> keep AND x <> dup
MERGE (keep)-[n:{{rel}}]->(x)
ON CREATE SET n.tenant_id = $tenant_id, n.created_at = datetime()
SET n.chunk_ids = {_UNION.format(a="n.chunk_ids", b="r.chunk_ids")},
    n.document_ids = {_UNION.format(a="n.document_ids", b="r.document_ids")},
    n.evidence = coalesce(n.evidence, r.evidence),
    n.manual = coalesce(n.manual, false) OR coalesce(r.manual, false)
DELETE r
"""

MOVE_INCOMING = f"""
MATCH (x:Entity {{tenant_id: $tenant_id}})-[r:{{rel}}]->(dup:Entity {{id: $dup_id, tenant_id: $tenant_id}})
MATCH (keep:Entity {{id: $keep_id, tenant_id: $tenant_id}})
WITH keep, dup, r, x WHERE x <> keep AND x <> dup
MERGE (x)-[n:{{rel}}]->(keep)
ON CREATE SET n.tenant_id = $tenant_id, n.created_at = datetime()
SET n.chunk_ids = {_UNION.format(a="n.chunk_ids", b="r.chunk_ids")},
    n.document_ids = {_UNION.format(a="n.document_ids", b="r.document_ids")},
    n.evidence = coalesce(n.evidence, r.evidence),
    n.manual = coalesce(n.manual, false) OR coalesce(r.manual, false)
DELETE r
"""

MOVE_MENTIONS = """
MATCH (c:Chunk {tenant_id: $tenant_id})-[m:MENTIONS]->(:Entity {id: $dup_id, tenant_id: $tenant_id})
MATCH (keep:Entity {id: $keep_id, tenant_id: $tenant_id})
MERGE (c)-[:MENTIONS]->(keep)
DELETE m
"""

ABSORB = f"""
MATCH (dup:Entity {{id: $dup_id, tenant_id: $tenant_id}})
MATCH (e:Entity {{id: $keep_id, tenant_id: $tenant_id}})
SET e.document_ids = {_UNION.format(a="e.document_ids", b="dup.document_ids")},
    e.aliases = ({_UNION.format(a="e.aliases", b="coalesce(dup.aliases, []) + dup.name")})[..50],
    e.merged_keys = {_UNION.format(a="e.merged_keys", b="[dup.type + '::' + dup.normalized_name] + coalesce(dup.merged_keys, [])")},
    e.description = CASE WHEN coalesce(e.description, '') = '' THEN dup.description ELSE e.description END,
    e.edited = true, e.updated_at = datetime()
SET e.aliases_text = {_ALIASES_TEXT}
DETACH DELETE dup
"""

DELETE_ENTITY = """
MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
WITH e, e.name AS name
DETACH DELETE e
RETURN name
"""

ADD_RELATIONSHIP = """
MATCH (s:Entity {id: $source_id, tenant_id: $tenant_id})
MATCH (t:Entity {id: $target_id, tenant_id: $tenant_id})
MERGE (s)-[r:{rel}]->(t)
ON CREATE SET r.tenant_id = $tenant_id, r.chunk_ids = [], r.document_ids = [], r.created_at = datetime()
SET r.manual = true, r.evidence = coalesce($evidence, r.evidence), r.updated_at = datetime()
SET s.edited = true, t.edited = true  // curated endpoints survive re-ingestion with the edge
RETURN s.name AS source, s.type AS source_type, type(r) AS relationship, t.name AS target, t.type AS target_type,
       s.id AS source_id, t.id AS target_id, r.evidence AS evidence, true AS manual
"""

DELETE_RELATIONSHIP = """
MATCH (:Entity {id: $source_id, tenant_id: $tenant_id})-[r:{rel}]->(:Entity {id: $target_id, tenant_id: $tenant_id})
DELETE r
RETURN count(*) AS deleted
"""


def _dedupe(values: list[str], limit: int = 50) -> list[str]:
    seen: dict[str, str] = {}
    for v in values:
        v = " ".join(str(v).split())
        if v and v.lower() not in seen:
            seen[v.lower()] = v
    return list(seen.values())[:limit]


class GraphEditor:
    def __init__(self, driver: AsyncDriver, settings: Settings) -> None:
        self.driver = driver
        self.db = settings.neo4j_database

    async def _tx(self, work: Any) -> Any:
        try:
            async with self.driver.session(database=self.db) as session:
                return await session.execute_write(work)
        except (ServiceUnavailable, SessionExpired) as exc:
            raise Neo4jUnavailable() from exc

    @staticmethod
    async def _rows(tx: AsyncManagedTransaction, cypher: str, **params: Any) -> list[dict[str, Any]]:
        result = await tx.run(cypher, params)
        return [r.data() async for r in result]

    async def _get(self, tx: AsyncManagedTransaction, tenant_id: str, entity_id: str) -> dict[str, Any]:
        rows = await self._rows(tx, GET_ENTITY, tenant_id=tenant_id, entity_id=entity_id)
        if not rows:
            raise NotFoundError("Entity not found", code="ENTITY_NOT_FOUND")
        return rows[0]

    # ------------------------------------------------------------- entities
    async def update_entity(self, tenant_id: str, entity_id: str, *, name: str | None = None,
                            entity_type: str | None = None, description: str | None = None,
                            aliases: list[str] | None = None) -> dict[str, Any]:
        tenant_id = _require_tenant(tenant_id)
        label = safe_entity_label(entity_type) if entity_type else None

        async def work(tx: AsyncManagedTransaction) -> dict[str, Any]:
            current = await self._get(tx, tenant_id, entity_id)
            new_name = " ".join(name.split()) if name else current["name"]
            new_type = label or current["type"]
            new_key = canonical_key(new_name, new_type)
            if not new_key:
                raise ValidationFailed("The entity name is empty after normalisation")
            dup = await self._rows(tx, FIND_DUPLICATE, tenant_id=tenant_id, type=new_type,
                                   normalized_name=new_key, entity_id=entity_id)
            if dup:
                raise ConflictError(f"A {new_type} named '{dup[0]['name']}' already exists - merge the entities instead",
                                    code="ENTITY_EXISTS", details={"existing_id": dup[0]["id"]})
            old_key = f"{current['type']}::{current['normalized_name']}"
            merged = list(current["merged_keys"])
            if old_key != f"{new_type}::{new_key}" and old_key not in merged:
                merged.append(old_key)
            new_aliases = list(aliases) if aliases is not None else list(current["aliases"])
            if new_name != current["name"]:
                new_aliases.append(current["name"])
            await self._rows(
                tx, UPDATE_ENTITY.replace("{label}", new_type), tenant_id=tenant_id, entity_id=entity_id,
                name=new_name, normalized_name=new_key, type=new_type,
                description=current["description"] if description is None else description.strip(),
                aliases=_dedupe([a for a in new_aliases if a.lower() != new_name.lower()]), merged_keys=merged,
            )
            return await self._get(tx, tenant_id, entity_id)

        return await self._tx(work)

    async def merge_entities(self, tenant_id: str, keep_id: str, merge_ids: list[str]) -> dict[str, Any]:
        tenant_id = _require_tenant(tenant_id)
        merge_ids = [m for m in dict.fromkeys(merge_ids) if m != keep_id]
        if not merge_ids:
            raise ValidationFailed("Select at least one other entity to merge")

        async def work(tx: AsyncManagedTransaction) -> dict[str, Any]:
            await self._get(tx, tenant_id, keep_id)
            for dup_id in merge_ids:
                await self._get(tx, tenant_id, dup_id)
                params = {"tenant_id": tenant_id, "keep_id": keep_id, "dup_id": dup_id}
                for rel in RelationType:
                    await self._rows(tx, MOVE_OUTGOING.replace("{rel}", rel.value), **params)
                    await self._rows(tx, MOVE_INCOMING.replace("{rel}", rel.value), **params)
                await self._rows(tx, MOVE_MENTIONS, **params)
                await self._rows(tx, ABSORB, **params)
            return await self._get(tx, tenant_id, keep_id)

        return await self._tx(work)

    async def delete_entity(self, tenant_id: str, entity_id: str) -> str:
        tenant_id = _require_tenant(tenant_id)

        async def work(tx: AsyncManagedTransaction) -> str:
            rows = await self._rows(tx, DELETE_ENTITY, tenant_id=tenant_id, entity_id=entity_id)
            if not rows:
                raise NotFoundError("Entity not found", code="ENTITY_NOT_FOUND")
            return rows[0]["name"]

        return await self._tx(work)

    # -------------------------------------------------------- relationships
    async def add_relationship(self, tenant_id: str, source_id: str, rel_type: str, target_id: str,
                               evidence: str | None = None) -> dict[str, Any]:
        tenant_id = _require_tenant(tenant_id)
        rel = safe_rel_type(rel_type)
        if rel not in {r.value for r in RelationType}:
            raise ValidationFailed(f"'{rel}' is a structural relationship and cannot be added by hand")
        if source_id == target_id:
            raise ValidationFailed("A relationship needs two different entities")

        async def work(tx: AsyncManagedTransaction) -> dict[str, Any]:
            source = await self._get(tx, tenant_id, source_id)
            target = await self._get(tx, tenant_id, target_id)
            if not is_valid_relation(rel, source["type"], target["type"]):
                raise ValidationFailed(
                    f"{source['type']} -[{rel}]-> {target['type']} is not allowed by the graph schema",
                    code="INVALID_RELATIONSHIP")
            rows = await self._rows(tx, ADD_RELATIONSHIP.replace("{rel}", rel), tenant_id=tenant_id,
                                    source_id=source_id, target_id=target_id, evidence=(evidence or "").strip() or None)
            return rows[0]

        return await self._tx(work)

    async def delete_relationship(self, tenant_id: str, source_id: str, rel_type: str, target_id: str) -> None:
        tenant_id = _require_tenant(tenant_id)
        rel = safe_rel_type(rel_type)

        async def work(tx: AsyncManagedTransaction) -> int:
            rows = await self._rows(tx, DELETE_RELATIONSHIP.replace("{rel}", rel), tenant_id=tenant_id,
                                    source_id=source_id, target_id=target_id)
            return rows[0]["deleted"] if rows else 0

        if not await self._tx(work):
            raise NotFoundError("Relationship not found", code="RELATIONSHIP_NOT_FOUND")
