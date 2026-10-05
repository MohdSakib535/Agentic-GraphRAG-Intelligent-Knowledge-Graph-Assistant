from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MetadataFilter(BaseModel):
    document_ids: list[str] | None = Field(default=None, max_length=50)
    filenames: list[str] | None = Field(default=None, max_length=50)
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    section: str | None = Field(default=None, max_length=200)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: Literal["VECTOR", "GRAPH", "HYBRID"] = "HYBRID"
    top_k: int = Field(default=8, ge=1, le=50)
    filters: MetadataFilter | None = None
    rerank: bool = True

    model_config = ConfigDict(
        json_schema_extra={"example": {"query": "Which projects use Kafka?", "strategy": "HYBRID", "top_k": 5}}
    )


class ChunkHit(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    source_filename: str | None = None
    page_number: int | None = None
    section: str | None = None
    retrievers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphFact(BaseModel):
    source_id: str | None = None
    target_id: str | None = None
    source: str
    source_type: str
    relationship: str
    target: str
    target_type: str
    evidence: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    score: float = 1.0
    hops: int = 1
    manual: bool = False


class SearchResponse(BaseModel):
    query: str
    strategy: str
    chunks: list[ChunkHit]
    facts: list[GraphFact]
    linked_entities: list[dict[str, Any]] = Field(default_factory=list)
    cypher: str | None = None
    latency_ms: int
    cached: bool = False


class EntityOut(BaseModel):
    id: str
    name: str
    type: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    degree: int = 0
    document_ids: list[str] = Field(default_factory=list)


class EntityDetail(BaseModel):
    entity: EntityOut
    relationships: list[GraphFact]
    neighbors: list[EntityOut]
    sources: list[dict[str, Any]]


class SubgraphOut(BaseModel):
    nodes: list[EntityOut]
    edges: list[GraphFact]


class GraphStats(BaseModel):
    entities: int
    relationships: int
    chunks: int
    documents: int
    entities_by_type: dict[str, int]
    relationships_by_type: dict[str, int]


# ------------------------------------------------------------------ graph curation (admin)
class EntityUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    type: str | None = Field(default=None, max_length=40)
    description: str | None = Field(default=None, max_length=2000)
    aliases: list[str] | None = Field(default=None, max_length=50)


class EntityMerge(BaseModel):
    keep_id: str = Field(min_length=1, max_length=100, description="Entity that survives")
    merge_ids: list[str] = Field(min_length=1, max_length=20, description="Entities folded into keep_id")


class RelationshipCreate(BaseModel):
    source_id: str = Field(min_length=1, max_length=100)
    type: str = Field(min_length=1, max_length=40)
    target_id: str = Field(min_length=1, max_length=100)
    evidence: str | None = Field(default=None, max_length=1000)


class RelationshipRef(BaseModel):
    source_id: str = Field(min_length=1, max_length=100)
    type: str = Field(min_length=1, max_length=40)
    target_id: str = Field(min_length=1, max_length=100)


class EditedEntity(BaseModel):
    id: str
    name: str
    type: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
