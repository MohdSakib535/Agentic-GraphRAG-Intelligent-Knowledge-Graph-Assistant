"""Retrieval result models (JSON-serialisable so they can be cached in Redis)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.search import ChunkHit, GraphFact


class LinkedEntity(BaseModel):
    id: str
    name: str
    type: str
    description: str | None = None
    query: str | None = None
    score: float = 1.0


class RetrievalResult(BaseModel):
    strategy: str
    query: str
    chunks: list[ChunkHit] = Field(default_factory=list)
    facts: list[GraphFact] = Field(default_factory=list)
    linked_entities: list[LinkedEntity] = Field(default_factory=list)
    bridges: list[LinkedEntity] = Field(default_factory=list)
    answer_candidates: list[LinkedEntity] = Field(default_factory=list)
    cypher: str | None = None
    cypher_rows: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = 0
    cached: bool = False
    errors: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunks and not self.facts and not self.cypher_rows
