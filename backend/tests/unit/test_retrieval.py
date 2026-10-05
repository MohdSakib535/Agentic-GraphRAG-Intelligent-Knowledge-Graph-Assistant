"""Vector, graph and hybrid retrieval over the real ingested sample corpus (in-memory graph store)."""

from __future__ import annotations

from app.core.access import UNRESTRICTED, set_scope
from conftest import TENANT_A, TENANT_B


async def test_ingestion_builds_deduplicated_graph(sample_graph) -> None:
    set_scope(UNRESTRICTED)
    names = {(e["name"], e["type"]) for e in sample_graph.entities.values() if e["tenant_id"] == TENANT_A}
    for expected in [("Rahul", "Person"), ("Amit", "Person"), ("Priya", "Person"), ("Neha", "Person"),
                     ("Project Alpha", "Project"), ("Project Beta", "Project"), ("Project Gamma", "Project"),
                     ("Kafka", "Technology"), ("Redis", "Technology"), ("PostgreSQL", "Technology"),
                     ("FastAPI", "Technology"), ("Django", "Technology"), ("Neo4j", "Technology"), ("TechCorp", "Company")]:
        assert expected in names
    kafka = [e for e in sample_graph.entities.values() if e["tenant_id"] == TENANT_A and e["normalized_name"] == "kafka"]
    assert len(kafka) == 1 and len(kafka[0]["document_ids"]) >= 3  # one node, merged across documents
    chunks = [c for c in sample_graph.chunks.values() if c["tenant_id"] == TENANT_A]
    assert chunks and all(len(c["embedding"]) == 256 for c in chunks)


async def test_vector_similarity_search(container) -> None:
    set_scope(UNRESTRICTED)
    hits = await container.retrieval.vector.similarity_search("What is Kafka?", TENANT_A, top_k=5)
    assert hits and "Kafka" in hits[0].text
    assert hits[0].source_filename and hits[0].score > 0
    filtered = await container.retrieval.vector.similarity_search(
        "What is Kafka?", TENANT_A, top_k=5, filters={"filenames": ["technology-glossary.txt"]})
    assert filtered and all(h.source_filename == "technology-glossary.txt" for h in filtered)


async def test_graph_search_single_hop(container) -> None:
    set_scope(UNRESTRICTED)
    result = await container.retrieval.graph.search("Who manages Project Alpha?", TENANT_A)
    assert [c.name for c in result.answer_candidates] == ["Rahul"]
    assert any(f.source == "Rahul" and f.relationship == "MANAGES" and f.target == "Project Alpha" and f.chunk_ids
               for f in result.facts)


async def test_graph_search_multi_hop_traversal(container) -> None:
    set_scope(UNRESTRICTED)
    result = await container.retrieval.graph.search("Who works on projects managed by Rahul that use Kafka?", TENANT_A)
    assert {b.name for b in result.bridges} == {"Project Alpha"}
    assert sorted(c.name for c in result.answer_candidates) == ["Amit", "Neha"]
    two_hop = await container.retrieval.graph.search("Who manages projects that use Kafka?", TENANT_A)
    assert sorted(c.name for c in two_hop.answer_candidates) == ["Priya", "Rahul"]


async def test_hybrid_search_fuses_graph_and_vector(container) -> None:
    set_scope(UNRESTRICTED)
    result = await container.retrieval.retrieve("HYBRID", "Which developers work on Kafka projects managed by Rahul?",
                                                TENANT_A, relations=["WORKS_ON", "MANAGES"], answer_type="Person")
    assert result.strategy == "HYBRID" and result.chunks and result.facts
    assert any("graph_evidence" in h.retrievers for h in result.chunks)
    assert sorted(c.name for c in result.answer_candidates) == ["Amit", "Neha"]


async def test_retrieval_never_crosses_tenants(container) -> None:
    set_scope(UNRESTRICTED)
    hits = await container.retrieval.vector.similarity_search("Project Zeta Cassandra", TENANT_A, top_k=20)
    assert all("Zeta" not in h.text for h in hits)
    zeta = await container.retrieval.graph.search("Who manages Project Zeta?", TENANT_A)
    assert not zeta.facts and not zeta.linked_entities
    b_side = await container.retrieval.retrieve("HYBRID", "Who manages Project Alpha?", TENANT_B)
    assert all("Alpha" not in h.text for h in b_side.chunks) and not b_side.facts
    own = await container.retrieval.graph.search("Who manages Project Zeta?", TENANT_B)
    assert [c.name for c in own.answer_candidates] == ["Zed"]
