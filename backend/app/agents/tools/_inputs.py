from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.graph.schema import ENTITY_TYPES, RelationType
from app.schemas.search import MetadataFilter


class RetrievalToolInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Natural-language search query")
    entities: list[str] = Field(default_factory=list, max_length=15, description="Named entities in the query")
    relations: list[str] = Field(default_factory=list, max_length=8, description="Relationship types of interest")
    answer_type: str | None = Field(default=None, description="Expected entity type of the answer")
    top_k: int = Field(default=8, ge=1, le=50)
    filters: MetadataFilter | None = None

    @field_validator("relations")
    @classmethod
    def _rels(cls, value: list[str]) -> list[str]:
        allowed = {r.value for r in RelationType}
        return [v.upper() for v in value if v.upper() in allowed]

    @field_validator("answer_type")
    @classmethod
    def _atype(cls, value: str | None) -> str | None:
        return value if value in ENTITY_TYPES else None


def as_filter_dict(filters: object) -> dict[str, object] | None:
    if filters is None:
        return None
    if isinstance(filters, MetadataFilter):
        data = filters.model_dump(exclude_none=True)
    elif isinstance(filters, dict):
        data = MetadataFilter.model_validate(filters).model_dump(exclude_none=True)
    else:
        return None
    return data or None
