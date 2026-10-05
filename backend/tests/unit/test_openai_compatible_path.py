"""Exercise the OpenAI-compatible code paths (ChatOpenAI structured output + streaming, OpenAIEmbeddings)
against a local OpenAI-compatible stub server - no network, no API key."""

from __future__ import annotations

import os
import uuid

import pytest
from pydantic import SecretStr

from app.agents.workflow import RECURSION_LIMIT, initial_turn_state, thread_id
from app.core.config import Settings
from app.ingestion.embedding import OpenAIEmbedder
from app.ingestion.relationship_extractor import LLMGraphExtractor, build_graph_extractor
from app.llm.client import LLMClient, start_usage_tracking
from conftest import TENANT_A
from openai_stub import StubServer


@pytest.fixture(scope="module")
def stub():
    with StubServer(dims=256) as server:
        yield server


@pytest.fixture
def llm_settings(stub, test_settings) -> Settings:
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        os.environ.pop(var, None)  # the stub is local
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    return test_settings.model_copy(update={
        "llm_provider": "openai", "embedding_provider": "openai", "openai_api_key": SecretStr("sk-test-stub"),
        "openai_base_url": stub.base_url, "embedding_model": "text-embedding-3-small", "embedding_dimensions": 256,
        "llm_timeout_seconds": 20, "enable_text2cypher": True,
    })


def test_openai_embeddings_are_batched_and_dimension_checked(llm_settings, stub) -> None:
    embedder = OpenAIEmbedder(llm_settings)
    vectors = embedder.embed_documents([f"text {i}" for i in range(130)])
    assert len(vectors) == 130 and all(len(v) == 256 for v in vectors)
    batches = [r for r in stub.app.state.requests if r["kind"] == "embeddings"]
    assert batches[-1]["dimensions"] == 256 and max(b["n"] for b in batches) <= llm_settings.embedding_batch_size


def test_llm_graph_extraction_validates_structured_output(llm_settings) -> None:
    extractor = build_graph_extractor(llm_settings, LLMClient(llm_settings))
    assert isinstance(extractor, LLMGraphExtractor)
    result = extractor.extract("Rahul manages Project Alpha.", "c1")
    assert {e.name for e in result.entities} == {"Rahul", "Project Alpha"}  # 'Starship' entity rejected
    assert [(r.source, r.relationship.value, r.target) for r in result.relationships] == [("Rahul", "MANAGES", "Project Alpha")]


async def test_agent_runs_on_openai_compatible_llm(llm_settings, sample_graph, stub) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from app.core.container import build_container
    from app.ingestion.embedding import HashingEmbedder

    container = build_container(llm_settings, None, None, checkpointer=InMemorySaver(), reader=sample_graph,
                                embedder=HashingEmbedder(256))  # corpus was indexed with hashing embeddings
    assert container.llm is not None
    usage = start_usage_tracking()
    conversation = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id(TENANT_A, conversation), "tenant_id": TENANT_A, "denied_document_ids": []},
              "recursion_limit": RECURSION_LIMIT}
    state, tokens = {}, []
    async for mode, chunk in container.agent.astream(
        initial_turn_state("Who manages Project Alpha?", TENANT_A, conversation, "t"), config=config,
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            if chunk["event"] == "token":
                tokens.append(chunk["data"]["text"])
            continue
        for node, update in chunk.items():
            if node != "finalize":
                state.update(update)
    assert state["retrieval_strategy"] == "GRAPH"
    assert "Rahul manages Project Alpha" in state["answer"] and "[" in state["answer"]
    assert "".join(tokens).strip() == state["answer"]  # answer was streamed token by token
    assert state["verification"]["method"] == "llm+heuristic" and state["verification"]["passed"]
    schemas = {r["schema"] for r in stub.app.state.requests if r["kind"] == "chat"}
    assert {"QueryAnalysis", "LLMVerification", None} <= schemas  # analysis, verification, streamed generation
    assert usage.total_tokens > 0 and usage.calls >= 3  # token usage is tracked from API usage metadata
