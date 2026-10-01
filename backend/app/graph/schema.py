"""Knowledge-graph schema: the single source of truth for labels and relationship types.

Relationship types are a closed set. Cypher cannot parameterise labels or
relationship types, so any code that interpolates one into a query MUST go through
:func:`safe_rel_type` / :func:`safe_entity_label`, which only return values from
these whitelists.
"""

from __future__ import annotations

import enum

from app.core.errors import ValidationFailed


class EntityType(enum.StrEnum):
    PERSON = "Person"
    COMPANY = "Company"
    PROJECT = "Project"
    TECHNOLOGY = "Technology"
    DEPARTMENT = "Department"
    PRODUCT = "Product"
    LOCATION = "Location"
    CONCEPT = "Concept"


class RelationType(enum.StrEnum):
    WORKS_FOR = "WORKS_FOR"
    WORKS_ON = "WORKS_ON"
    MANAGES = "MANAGES"
    USES = "USES"
    BELONGS_TO = "BELONGS_TO"
    REPORTS_TO = "REPORTS_TO"
    DEPENDS_ON = "DEPENDS_ON"
    RELATED_TO = "RELATED_TO"


# Structural relationships created by the pipeline (not extracted from text).
STRUCTURAL_RELATIONS = ("CONTAINS", "MENTIONS")
ALL_RELATION_TYPES = tuple(r.value for r in RelationType) + STRUCTURAL_RELATIONS
ENTITY_TYPES = tuple(e.value for e in EntityType)
NODE_LABELS = (*ENTITY_TYPES, "Entity", "Document", "Chunk")

_ANY = frozenset(ENTITY_TYPES)
P, C, PR, T, D, PD, L, CO = (e.value for e in EntityType)

# (allowed source types, allowed target types) per relationship.
RELATION_CONSTRAINTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    RelationType.WORKS_FOR: (frozenset({P}), frozenset({C, D})),
    RelationType.WORKS_ON: (frozenset({P, D, C}), frozenset({PR, PD})),
    RelationType.MANAGES: (frozenset({P, D}), frozenset({PR, PD, D, P})),
    RelationType.USES: (frozenset({PR, PD, C, D, P}), frozenset({T, PD})),
    RelationType.BELONGS_TO: (_ANY, frozenset({C, D, PR, L, CO})),
    RelationType.REPORTS_TO: (frozenset({P, D}), frozenset({P, D})),
    RelationType.DEPENDS_ON: (frozenset({PR, PD, T}), frozenset({PR, PD, T})),
    RelationType.RELATED_TO: (_ANY, _ANY),
}

RELATION_DESCRIPTIONS: dict[str, str] = {
    RelationType.WORKS_FOR: "(Person)-[:WORKS_FOR]->(Company|Department)",
    RelationType.WORKS_ON: "(Person)-[:WORKS_ON]->(Project|Product)",
    RelationType.MANAGES: "(Person)-[:MANAGES]->(Project|Product|Department|Person)",
    RelationType.USES: "(Project|Product|Company|Department|Person)-[:USES]->(Technology|Product)",
    RelationType.BELONGS_TO: "(Any)-[:BELONGS_TO]->(Company|Department|Project|Location|Concept)",
    RelationType.REPORTS_TO: "(Person)-[:REPORTS_TO]->(Person)",
    RelationType.DEPENDS_ON: "(Project|Product|Technology)-[:DEPENDS_ON]->(Project|Product|Technology)",
    RelationType.RELATED_TO: "(Any)-[:RELATED_TO]->(Any)",
}


def is_valid_relation(rel: str, source_type: str, target_type: str) -> bool:
    constraint = RELATION_CONSTRAINTS.get(rel)
    if constraint is None:
        return False
    sources, targets = constraint
    return source_type in sources and target_type in targets


def safe_rel_type(value: str) -> str:
    upper = value.strip().upper()
    if upper not in ALL_RELATION_TYPES:
        raise ValidationFailed(f"Unsupported relationship type: {value!r}")
    return upper


def safe_entity_label(value: str) -> str:
    for label in ENTITY_TYPES:
        if label.lower() == value.strip().lower():
            return label
    raise ValidationFailed(f"Unsupported entity type: {value!r}")


VECTOR_INDEX_NAME = "chunk_embedding_index"
CHUNK_FULLTEXT_INDEX = "chunk_text_fulltext"
ENTITY_FULLTEXT_INDEX = "entity_name_fulltext"


def schema_statements(embedding_dimensions: int) -> list[str]:
    """Idempotent DDL creating constraints and indexes (vector + full-text)."""
    dims = int(embedding_dimensions)
    return [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)",
        "CREATE INDEX entity_tenant_norm IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.normalized_name)",
        "CREATE INDEX entity_tenant_type IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.type)",
        "CREATE INDEX chunk_tenant IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id)",
        "CREATE INDEX chunk_tenant_document IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id, c.document_id)",
        "CREATE INDEX document_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant_id)",
        (
            f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, `vector.similarity_function`: 'cosine'}}}}"
        ),
        f"CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
        f"CREATE FULLTEXT INDEX {ENTITY_FULLTEXT_INDEX} IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases_text]",
    ]


def schema_prompt() -> str:
    """Compact schema description used by schema-aware Text2Cypher prompts."""
    rels = "\n".join(f"- {d}" for d in RELATION_DESCRIPTIONS.values())
    return (
        "Node labels (every entity node also has label :Entity): "
        + ", ".join(ENTITY_TYPES)
        + ".\nEntity properties: name (string), type (string), description (string), normalized_name.\n"
        "Chunk properties: text, document_id, page_number, section, source_filename. "
        "(:Document)-[:CONTAINS]->(:Chunk)-[:MENTIONS]->(:Entity).\n"
        f"Relationships:\n{rels}\n"
        "Relationship properties: chunk_ids (list), document_ids (list), evidence (string)."
    )
