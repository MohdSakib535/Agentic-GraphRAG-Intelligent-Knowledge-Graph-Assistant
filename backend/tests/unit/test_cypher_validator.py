from __future__ import annotations

import pytest

from app.core.errors import CypherValidationError
from app.graph.cypher_validator import validate_cypher

# (query, number of node patterns that must receive the tenant constraint)
SAFE = [
    ("MATCH (p:Person)-[:MANAGES]->(proj:Project)-[:USES]->(t:Technology {name: 'Kafka'}) RETURN p.name, proj.name", 3),
    ("MATCH (p:Person) WHERE toLower(p.name) CONTAINS 'rahul' OPTIONAL MATCH (p)-[r]->(x) RETURN p.name, type(r), x.name", 3),
    ("MATCH (a:Project), (b:Project) WITH a, b MATCH (a)-[:USES]->(t)<-[:USES]-(b) RETURN DISTINCT t.name ORDER BY t.name", 5),
    ("UNWIND ['Kafka', 'Redis'] AS n MATCH (t:Technology {name: n}) RETURN t.name, count(*) AS c", 1),
    ("MATCH path = shortestPath((a:Person)-[*..3]-(b:Technology)) RETURN path", 2),
    ("MATCH (p:Person) RETURN p.name, p.role LIMIT 10", 1),
]

DANGEROUS = [
    "MATCH (n) DETACH DELETE n",
    "MATCH (n) DELETE n",
    "CREATE (n:Person {name: 'x'}) RETURN n",
    "MERGE (n:Person {name: 'x'}) RETURN n",
    "MATCH (n:Person) SET n.name = 'x' RETURN n",
    "MATCH (n:Person) REMOVE n.name RETURN n",
    "DROP INDEX chunk_embedding_index",
    "CALL db.labels() YIELD label RETURN label",
    "MATCH (n) RETURN apoc.convert.toJson(n)",
    "LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row",
    "MATCH (n) RETURN n; MATCH (m) DETACH DELETE m",
    "MATCH (n) RETURN n // CREATE (x)",
    "MATCH (n) WHERE EXISTS { MATCH (n)-->(m) } RETURN n",
    "MATCH (n:Person) WHERE n.tenant_id = 'another-tenant' RETURN n",
    "MATCH (n:Person) RETURN n, $secret",
    "MATCH (a)-[*]-(b) RETURN a",
    "MATCH (a)-[*1..20]-(b) RETURN a",
    "MATCH (n:Secret) RETURN n",
    "MATCH (n:Person)-[:OWNS_SERVER]->(m) RETURN n",
    "MATCH (`n`) RETURN n",
    "MATCH (n:Person WHERE n.name = 'x') RETURN n",
    "MATCH (n {meta: {a: 1}}) RETURN n",
    "FOREACH (x IN [1] | CREATE (:Person))",
    "SHOW DATABASES",
    "RETURN 1",
]


@pytest.mark.parametrize(("query", "node_patterns"), SAFE)
def test_safe_queries_are_scoped_to_tenant(query: str, node_patterns: int) -> None:
    validated = validate_cypher(query)
    assert validated.query.count("tenant_id: $tenant_id") == node_patterns  # every node pattern is scoped
    assert "LIMIT" in validated.query


@pytest.mark.parametrize("query", DANGEROUS)
def test_dangerous_or_unscoped_queries_are_rejected(query: str) -> None:
    with pytest.raises(CypherValidationError):
        validate_cypher(query)


def test_limit_is_enforced_and_strings_preserved() -> None:
    q = validate_cypher("MATCH (p:Person {name: 'DELETE me (please)'}) RETURN p.name LIMIT 5000", max_limit=50)
    assert q.query.endswith("LIMIT 50")
    assert "'DELETE me (please)'" in q.query  # keywords inside string literals are data, not code
    assert "(p:Person {name: 'DELETE me (please)', tenant_id: $tenant_id})" in q.query
