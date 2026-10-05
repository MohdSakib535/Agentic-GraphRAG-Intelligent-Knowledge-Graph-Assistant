"""Knowledge-graph curation (admin): fix, merge and delete entities and relationships."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from app.core.dependencies import AdminDep, ContainerDep, DBSession, SettingsDep
from app.graph.editor import GraphEditor
from app.graph.schema import RELATION_DESCRIPTIONS
from app.schemas.common import ERROR_RESPONSES
from app.schemas.search import EditedEntity, EntityMerge, EntityUpdate, GraphFact, RelationshipCreate, RelationshipRef
from app.services.audit import record_audit

router = APIRouter(prefix="/graph", tags=["Graph curation"], responses=ERROR_RESPONSES)


def _editor(container: ContainerDep, settings: SettingsDep) -> GraphEditor:
    return GraphEditor(container.reader.driver, settings)


async def _done(db: DBSession, container: ContainerDep, admin: AdminDep, action: str, resource_id: str,
                details: dict[str, Any]) -> None:
    record_audit(db, action, tenant_id=admin.tenant_id, user_id=admin.id, resource_type="graph",
                 resource_id=resource_id[:80], details=details)
    await db.commit()
    if container.cache is not None:
        await container.cache.invalidate_tenant(admin.tenant)  # cached answers/retrievals saw the old graph


@router.get("/schema", summary="Relationship types and the entity types they connect")
async def graph_schema(_: AdminDep) -> dict[str, str]:
    return dict(RELATION_DESCRIPTIONS)


@router.patch("/entities/{entity_id}", response_model=EditedEntity, summary="Rename, re-type or describe an entity")
async def update_entity(entity_id: str, body: EntityUpdate, admin: AdminDep, db: DBSession, container: ContainerDep,
                        settings: SettingsDep) -> EditedEntity:
    """Renaming onto an existing entity of the same type is rejected with ``ENTITY_EXISTS`` - merge instead."""
    row = await _editor(container, settings).update_entity(
        admin.tenant, entity_id, name=body.name, entity_type=body.type, description=body.description,
        aliases=body.aliases)
    await _done(db, container, admin, "graph.entity.update", entity_id,
                {"fields": sorted(body.model_dump(exclude_none=True))})
    return EditedEntity.model_validate(row)


@router.post("/entities/merge", response_model=EditedEntity, summary="Merge duplicate entities into one")
async def merge_entities(body: EntityMerge, admin: AdminDep, db: DBSession, container: ContainerDep,
                         settings: SettingsDep) -> EditedEntity:
    """Relationships, chunk mentions, aliases and provenance move onto ``keep_id``; the others are deleted.
    Future ingestion maps the merged names onto the surviving entity."""
    row = await _editor(container, settings).merge_entities(admin.tenant, body.keep_id, body.merge_ids)
    await _done(db, container, admin, "graph.entity.merge", body.keep_id, {"merged": body.merge_ids})
    return EditedEntity.model_validate(row)


@router.delete("/entities/{entity_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete an entity")
async def delete_entity(entity_id: str, admin: AdminDep, db: DBSession, container: ContainerDep,
                        settings: SettingsDep) -> None:
    await _editor(container, settings).delete_entity(admin.tenant, entity_id)
    await _done(db, container, admin, "graph.entity.delete", entity_id, {})


@router.post("/relationships", response_model=GraphFact, status_code=status.HTTP_201_CREATED,
             summary="Add a relationship (validated against the graph schema)")
async def add_relationship(body: RelationshipCreate, admin: AdminDep, db: DBSession, container: ContainerDep,
                           settings: SettingsDep) -> GraphFact:
    row = await _editor(container, settings).add_relationship(
        admin.tenant, body.source_id, body.type, body.target_id, body.evidence)
    await _done(db, container, admin, "graph.relationship.add", body.source_id,
                {"type": row["relationship"], "target_id": body.target_id})
    return GraphFact.model_validate(row)


@router.post("/relationships/delete", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a relationship")
async def delete_relationship(body: RelationshipRef, admin: AdminDep, db: DBSession, container: ContainerDep,
                              settings: SettingsDep) -> None:
    await _editor(container, settings).delete_relationship(admin.tenant, body.source_id, body.type, body.target_id)
    await _done(db, container, admin, "graph.relationship.delete", body.source_id,
                {"type": body.type.upper(), "target_id": body.target_id})
