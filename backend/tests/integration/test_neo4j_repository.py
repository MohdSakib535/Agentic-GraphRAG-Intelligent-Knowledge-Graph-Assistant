"""Repository-level tests against a real Neo4j: writes, vector index, isolation, read-only Text2Cypher."""

from __future__ import annotations

import uuid

import pytest

from app.core.errors import RetrievalError
from app.graph.cypher_validator import validate_cypher
from app.graph.repository import GraphReader, GraphWriter


@pytest.fixture
async def repo():
    from neo4j import AsyncGraphDatabase, GraphDatabase

    from app.core.config import get_settings
    from app.db.neo4j import ensure_schema_sync

    settings = get_settings().model_copy(update={"embedding_dimensions": 1536})
    auth = (settings.neo4j_username, settings.neo4j_password.get_secret_value())
    sync_driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
    ensure_schema_sync(sync_driver, settings)
    async_driver = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=auth)
    yield GraphWriter(sync_driver, settings), GraphReader(async_driver, settings)
    await async_driver.close()
    sync_driver.close()


def _populate(writer: GraphWriter, tenant: str, project: str) -> str:
    doc = str(uuid.uuid4())
    writer.upsert_document(tenant, doc, "doc.md", "Doc", "md")
    cid = f"chk_{uuid.UUID(doc).hex}_00000"
    writer.write_chunks(tenant, doc, [{"id": cid, "tenant_id": tenant, "document_id": doc, "chunk_index": 0,
                                       "text": f"Rahul manages {project}.", "token_count": 5, "page_number": 1,
                                       "page_end": 1, "section": "S", "source_filename": "doc.md", "document_title": "Doc"}])
    writer.set_chunk_embeddings(tenant, [{"id": cid, "embedding": [1.0] + [0.0] * 1535}])
    people = f"p_{tenant}"
    proj = f"x_{tenant}"
    writer.upsert_entities(tenant, [
        {"id": people, "tenant_id": tenant, "name": "Rahul", "type": "Person", "normalized_name": "rahul",
         "description": "", "aliases": [], "document_id": doc},
        {"id": proj, "tenant_id": tenant, "name": project, "type": "Project", "normalized_name": project.lower(),
         "description": "", "aliases": [], "document_id": doc}])
    writer.write_mentions(tenant, [(cid, people), (cid, proj)])
    writer.upsert_relationships(tenant, doc, [{"source_id": people, "target_id": proj, "type": "MANAGES",
                                               "evidence": "Rahul manages it", "chunk_ids": [cid]}])
    return doc


async def test_repository_roundtrip_isolation_and_delete(repo) -> None:
    writer, reader = repo
    t1, t2 = str(uuid.uuid4()), str(uuid.uuid4())
    doc1 = _populate(writer, t1, "Project One")
    _populate(writer, t2, "Project Two")
    hits = await reader.vector_search(t1, [1.0] + [0.0] * 1535, top_k=10)
    assert [h["text"] for h in hits] == ["Rahul manages Project One."]
    facts = await reader.neighborhood(t1, [f"p_{t1}"], ["MANAGES"], 10)
    assert [(f["source"], f["target"]) for f in facts] == [("Rahul", "Project One")]
    assert await reader.neighborhood(t1, [f"p_{t2}"], None, 10) == []  # other tenant's id is invisible
    linked = await reader.link_entities_exact(t1, [{"raw": "Rahul", "key": "rahul", "lower": "rahul"}])
    assert [row["id"] for row in linked] == [f"p_{t1}"]
    writer.delete_document(t1, doc1)
    assert (await reader.stats(t1))["entities"] == 0
    assert (await reader.stats(t2))["entities"] == 2
    writer.delete_document(t2, next(iter([d["id"] for d in await _docs(reader, t2)])))


async def _docs(reader: GraphReader, tenant: str) -> list[dict]:
    return await reader._read("MATCH (d:Document {tenant_id: $tenant_id}) RETURN d.id AS id", tenant_id=tenant)


async def test_text2cypher_execution_is_read_only_and_tenant_scoped(repo) -> None:
    writer, reader = repo
    t1, t2 = str(uuid.uuid4()), str(uuid.uuid4())
    d1, d2 = _populate(writer, t1, "Project One"), _populate(writer, t2, "Project Two")
    q = validate_cypher("MATCH (p:Person)-[:MANAGES]->(x:Project) RETURN p.name AS person, x.name AS project")
    rows = await reader.run_validated_readonly(q.query, {}, t1, 50)
    assert rows == [{"person": "Rahul", "project": "Project One"}]
    # Even if a write slipped past validation, the READ transaction is refused by the server.
    with pytest.raises(RetrievalError):
        await reader.run_validated_readonly("CREATE (n:Person {tenant_id: $tenant_id}) RETURN n", {}, t1, 5)
    assert (await reader.stats(t1))["entities"] == 2
    writer.delete_document(t1, d1)
    writer.delete_document(t2, d2)
