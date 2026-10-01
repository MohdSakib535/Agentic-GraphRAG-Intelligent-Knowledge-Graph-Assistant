"""Shared fixtures.

Unit/agent tests run fully in-process: the real ingestion pipeline (heuristic
extractor + hashing embeddings) populates an in-memory graph that implements the
repository interface, and the real retrievers and LangGraph agent run on top.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from app.core.config import Settings  # noqa: E402
from app.graph.builder import GraphBuilder  # noqa: E402
from app.ingestion.embedding import HashingEmbedder  # noqa: E402
from app.ingestion.entity_resolver import EntityResolver  # noqa: E402
from app.ingestion.loader import FileStorage  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.ingestion.relationship_extractor import HeuristicGraphExtractor  # noqa: E402
from fakes import InMemoryGraph  # noqa: E402

SAMPLES = BACKEND / "data" / "samples"
TENANT_A = str(uuid.UUID(int=0xA))
TENANT_B = str(uuid.UUID(int=0xB))


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        environment="test",
        llm_provider="heuristic",
        embedding_provider="hashing",
        embedding_dimensions=256,
        upload_dir=str(tmp_path_factory.mktemp("uploads")),
        rate_limit_enabled=False,
        enable_text2cypher=False,
    )


def ingest_samples(graph: InMemoryGraph, settings: Settings, tenant_id: str, files: list[str]) -> dict[str, str]:
    storage = FileStorage(settings.upload_dir)
    embedder = HashingEmbedder(settings.embedding_dimensions)
    pipeline = IngestionPipeline(
        settings=settings, storage=storage, embedder=embedder, extractor_factory=HeuristicGraphExtractor,
        resolver=EntityResolver(settings, embedder=embedder, lookup=graph), builder=GraphBuilder(graph),  # type: ignore[arg-type]
    )
    ids = {}
    for name in files:
        path = SAMPLES / name
        doc_id = str(uuid.uuid4())
        stored = storage.save(tenant_id, doc_id, path.suffix.lstrip("."), path.read_bytes())
        pipeline.run(tenant_id=tenant_id, document_id=doc_id, filename=name, file_type=path.suffix.lstrip("."),
                     storage_path=stored)
        ids[name] = doc_id
    return ids


ALL_SAMPLES = ["architecture.pdf", "project-overview.docx", "team-directory.md", "technology-glossary.txt"]


@pytest.fixture(scope="session")
def sample_graph(test_settings: Settings) -> InMemoryGraph:
    """Tenant A owns the full sample corpus; tenant B owns only a private, unrelated document."""
    graph = InMemoryGraph()
    ingest_samples(graph, test_settings, TENANT_A, ALL_SAMPLES)
    secret = Path(test_settings.upload_dir) / "secret.md"
    secret.write_text("# Secret\n\nZed manages Project Zeta. Project Zeta uses Cassandra.\n")
    storage = FileStorage(test_settings.upload_dir)
    embedder = HashingEmbedder(test_settings.embedding_dimensions)
    pipeline = IngestionPipeline(
        settings=test_settings, storage=storage, embedder=embedder, extractor_factory=HeuristicGraphExtractor,
        resolver=EntityResolver(test_settings, embedder=embedder, lookup=graph), builder=GraphBuilder(graph),  # type: ignore[arg-type]
    )
    doc_id = str(uuid.uuid4())
    stored = storage.save(TENANT_B, doc_id, "md", secret.read_bytes())
    pipeline.run(tenant_id=TENANT_B, document_id=doc_id, filename="secret.md", file_type="md", storage_path=stored)
    return graph


@pytest.fixture
def container(sample_graph: InMemoryGraph, test_settings: Settings):
    from langgraph.checkpoint.memory import InMemorySaver

    from app.core.container import build_container

    return build_container(test_settings, None, None, checkpointer=InMemorySaver(),
                           embedder=HashingEmbedder(test_settings.embedding_dimensions), reader=sample_graph)
