"""Direct retrieval and knowledge-graph exploration endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.dependencies import ContainerDep, CurrentUserDep, RateLimit
from app.graph.schema import ENTITY_TYPES
from app.schemas.common import ERROR_RESPONSES
from app.schemas.search import EntityDetail, EntityOut, GraphStats, SearchRequest, SearchResponse, SubgraphOut
from app.services.search_service import SearchService

router = APIRouter(tags=["Search & Graph"], responses=ERROR_RESPONSES)
TypeFilter = Annotated[list[str] | None, Query(description=f"Entity types: {', '.join(ENTITY_TYPES)}")]


@router.post("/search", response_model=SearchResponse, dependencies=[Depends(RateLimit("chat"))],
             summary="Run vector, graph or hybrid retrieval directly (no agent)")
async def search(body: SearchRequest, user: CurrentUserDep, container: ContainerDep) -> SearchResponse:
    return await SearchService(container).search(user.tenant, body)


@router.get("/graph/stats", response_model=GraphStats, summary="Knowledge-graph statistics for my tenant")
async def graph_stats(user: CurrentUserDep, container: ContainerDep) -> GraphStats:
    return await SearchService(container).stats(user.tenant)


@router.get("/graph/entities", response_model=list[EntityOut], summary="Search entities")
async def graph_entities(
    user: CurrentUserDep, container: ContainerDep, q: Annotated[str | None, Query(max_length=200)] = None,
    types: TypeFilter = None, limit: Annotated[int, Query(ge=1, le=500)] = 100, offset: Annotated[int, Query(ge=0)] = 0,
) -> list[EntityOut]:
    return await SearchService(container).entities(user.tenant, q, types, limit, offset)


@router.get("/graph/entities/{entity_id}", response_model=EntityDetail, summary="Entity details, relationships and sources")
async def graph_entity(entity_id: str, user: CurrentUserDep, container: ContainerDep) -> EntityDetail:
    return await SearchService(container).entity_detail(user.tenant, entity_id)


@router.get("/graph/subgraph", response_model=SubgraphOut, summary="Subgraph for visualisation")
async def graph_subgraph(
    user: CurrentUserDep, container: ContainerDep, entity_id: str | None = None, types: TypeFilter = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 300, depth: Annotated[int, Query(ge=1, le=3)] = 1,
) -> SubgraphOut:
    return await SearchService(container).subgraph(user.tenant, entity_id, types, limit, depth)
