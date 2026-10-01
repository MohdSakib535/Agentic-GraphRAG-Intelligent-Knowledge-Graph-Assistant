from __future__ import annotations

from app.core.config import Settings
from app.graph.schema import is_valid_relation
from app.ingestion.entity_extractor import HeuristicEntityExtractor, validate_entities
from app.ingestion.entity_resolver import EntityResolver, canonical_key
from app.ingestion.relationship_extractor import (
    HeuristicGraphExtractor,
    LLMGraphExtractor,
    validate_relationships,
)
from fakes import FakeStructuredLLM

TEXT = (
    "Rahul Sharma is the engineering manager at TechCorp. Rahul manages Project Alpha and Project Beta. "
    "Project Alpha uses Apache Kafka, Redis and PostgreSQL. Amit, a senior backend developer, works on Project Alpha. "
    "Priya and Neha work on Project Beta, which is built with Django. Neha reports to Rahul. "
    "Project Gamma is managed by Priya. The Data Platform team is based in Bangalore."
)


def triples(text: str) -> set[tuple[str, str, str]]:
    extractor = HeuristicGraphExtractor()
    extractor.prime([text])
    result = extractor.extract(text, "c1")
    return {(r.source, r.relationship.value, r.target) for r in result.relationships}


def test_heuristic_entity_extraction_types() -> None:
    extractor = HeuristicEntityExtractor()
    extractor.prime([TEXT])
    found = {(e.name, e.type.value) for e in extractor.extract(TEXT)}
    expected = {("Rahul Sharma", "Person"), ("TechCorp", "Company"), ("Project Alpha", "Project"),
                ("Kafka", "Technology"), ("Amit", "Person"), ("Priya", "Person"), ("Neha", "Person"),
                ("Django", "Technology"), ("Data Platform Team", "Department"), ("Bangalore", "Location")}
    assert expected <= found


def test_heuristic_relationship_extraction() -> None:
    rels = triples(TEXT)
    assert ("Rahul Sharma", "MANAGES", "Project Alpha") in rels
    assert ("Rahul Sharma", "MANAGES", "Project Beta") in rels  # coordinated objects
    assert {("Project Alpha", "USES", t) for t in ("Kafka", "Redis", "PostgreSQL")} <= rels  # lists
    assert ("Priya", "WORKS_ON", "Project Beta") in rels and ("Neha", "WORKS_ON", "Project Beta") in rels
    assert ("Project Beta", "USES", "Django") in rels  # relative clause
    assert ("Priya", "MANAGES", "Project Gamma") in rels  # passive voice
    assert ("Neha", "REPORTS_TO", "Rahul Sharma") in rels  # first-name resolution
    assert all(is_valid_relation(r, _type(s), _type(t)) for s, r, t in rels)


def _type(name: str) -> str:
    return {"Rahul Sharma": "Person", "Amit": "Person", "Priya": "Person", "Neha": "Person", "TechCorp": "Company",
            "Bangalore": "Location", "Data Platform Team": "Department"}.get(
        name, "Project" if name.startswith("Project") else "Technology")


def test_validation_never_trusts_llm_output() -> None:
    entities = validate_entities([
        {"name": "Kafka", "type": "technology"},  # case-insensitive type coercion
        {"name": "Rahul", "type": "Person"},
        {"name": "Evil", "type": "Spaceship"},  # unknown type
        {"name": "<script>", "type": "Person"},  # forbidden characters
        {"name": "", "type": "Person"},
        "not even a dict",
        {"name": "Kafka", "type": "Technology"},  # duplicate
    ])
    assert [(e.name, e.type.value) for e in entities] == [("Kafka", "Technology"), ("Rahul", "Person")]
    rels = validate_relationships([
        {"source": "Rahul", "relationship": "uses", "target": "Kafka"},
        {"source": "Rahul", "relationship": "HACKS", "target": "Kafka"},  # not whitelisted
        {"source": "Kafka", "relationship": "MANAGES", "target": "Rahul"},  # violates type constraints
        {"source": "Rahul", "relationship": "USES", "target": "Unknown"},  # endpoint not extracted
    ], entities)
    assert [(r.source, r.relationship.value, r.target) for r in rels] == [("Rahul", "USES", "Kafka")]


def test_llm_extractor_validates_structured_output() -> None:
    llm = FakeStructuredLLM({"extract_graph": {
        "entities": [{"name": "Project Alpha", "type": "Project"}, {"name": "Kafka", "type": "Technology"},
                     {"name": "Bad", "type": "Alien"}],
        "relationships": [{"source": "Project Alpha", "relationship": "USES", "target": "Kafka"},
                          {"source": "Project Alpha", "relationship": "DESTROYS", "target": "Kafka"},
                          {"source": "Project Alpha", "relationship": "USES", "target": "Bad"}],
    }})
    result = LLMGraphExtractor(llm).extract("Project Alpha uses Kafka.", "c1")  # type: ignore[arg-type]
    assert {e.name for e in result.entities} == {"Project Alpha", "Kafka"}
    assert [(r.source, r.relationship.value, r.target) for r in result.relationships] == [("Project Alpha", "USES", "Kafka")]


def test_canonical_keys_resolve_aliases() -> None:
    keys = {canonical_key(n, "Technology") for n in ("Kafka", "Apache Kafka", "apache kafka", "Kafka platform")}
    assert keys == {"kafka"}
    assert canonical_key("Postgres", "Technology") == canonical_key("PostgreSQL database", "Technology") == "postgresql"
    assert canonical_key("TechCorp Inc.", "Company") == "techcorp"
    assert canonical_key("the Alpha project", "Project") == canonical_key("Project Alpha", "Project") == "project alpha"


class _Lookup:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def find_by_keys(self, tenant_id: str, keys: list[str]) -> list[dict]:
        return [r for r in self.rows if r["normalized_name"] in keys]

    def find_candidates(self, tenant_id: str, entity_type: str, tokens: list[str], limit: int) -> list[dict]:
        return [r for r in self.rows if r["type"] == entity_type and any(t in r["normalized_name"] for t in tokens)]


def test_entity_resolution_merges_duplicates_and_reuses_graph_entities() -> None:
    settings = Settings(environment="test", embedding_provider="hashing")
    text1 = "Rahul Sharma manages Project Alpha. Project Alpha uses Apache Kafka."
    text2 = "Rahul approved the plan. Amit works on the Alpha project, which relies on Kafka and Postgres."
    extractor = HeuristicGraphExtractor()
    extractor.prime([text1, text2])
    extractions = []
    for cid, text in (("c1", text1), ("c2", text2)):
        r = extractor.extract(text, cid)
        extractions.append((cid, r.entities, r.relationships))
    existing = [{"id": "ent_existing_pg", "name": "PostgreSQL", "type": "Technology", "normalized_name": "postgresql"}]
    result = EntityResolver(settings, lookup=_Lookup(existing)).resolve("tenant", extractions)
    names = sorted(e.name for e in result.entities)
    assert names == ["Amit", "Kafka", "PostgreSQL", "Project Alpha", "Rahul Sharma"]
    pg = next(e for e in result.entities if e.name == "PostgreSQL")
    assert pg.id == "ent_existing_pg" and pg.existing  # matched the graph instead of creating a duplicate
    by_id = {e.id: e.name for e in result.entities}
    rels = {(by_id[r.source_id], r.type, by_id[r.target_id]) for r in result.relationships}
    assert ("Project Alpha", "USES", "PostgreSQL") in rels and ("Amit", "WORKS_ON", "Project Alpha") in rels
    kafka_rel = next(r for r in result.relationships if by_id[r.target_id] == "Kafka")
    assert kafka_rel.chunk_ids == {"c1", "c2"}  # provenance from both chunks is kept
