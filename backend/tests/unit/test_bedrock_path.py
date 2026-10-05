"""Amazon Bedrock provider: one switch (USE_BEDROCK) moves the LLM and embeddings to Bedrock.

Runs the real langchain-aws / botocore clients (Converse, ConverseStream with AWS event-stream framing,
InvokeModel for Titan embeddings) against the local stub - no AWS account or network needed.
"""

from __future__ import annotations

import os
import uuid

import pytest
from pydantic import SecretStr

from app.agents.workflow import RECURSION_LIMIT, initial_turn_state, thread_id
from app.core.config import Settings
from app.ingestion.embedding import BedrockEmbedder, HashingEmbedder, OpenAIEmbedder, build_embedder
from app.ingestion.relationship_extractor import LLMGraphExtractor, build_graph_extractor
from app.llm.client import LLMClient, build_llm_client, message_text, start_usage_tracking
from conftest import TENANT_A
from openai_stub import StubServer

BEDROCK_KEY = "bedrock-api-key-test"


@pytest.fixture(scope="module")
def stub():
    with StubServer(dims=256) as server:
        yield server


@pytest.fixture
def bedrock_settings(stub, test_settings) -> Settings:
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    return Settings(**{
        **test_settings.model_dump(exclude={"llm_provider", "embedding_provider", "embedding_dimensions"}),
        "use_bedrock": True, "bedrock_api_key": SecretStr(BEDROCK_KEY), "aws_region": "us-east-1",
        "bedrock_endpoint_url": stub.bedrock_url, "bedrock_embedding_dimensions": 512,
        "openai_api_key": SecretStr("sk-also-configured"),  # USE_BEDROCK wins over the OpenAI key
        "llm_timeout_seconds": 20, "enable_text2cypher": True,
    })


def _requests(stub, kind: str) -> list[dict]:
    return [r for r in stub.app.state.requests if r["kind"] == kind]


def test_provider_switch_is_one_flag(test_settings) -> None:
    base = test_settings.model_dump(exclude={"llm_provider", "embedding_provider", "embedding_dimensions"})
    offline = Settings(**base)
    openai = Settings(**{**base, "openai_api_key": SecretStr("sk-x")})
    bedrock = Settings(**{**base, "openai_api_key": SecretStr("sk-x"), "use_bedrock": True})
    assert (offline.resolved_llm_provider, offline.resolved_embedding_provider) == ("heuristic", "hashing")
    assert (openai.resolved_llm_provider, openai.resolved_embedding_provider) == ("openai", "openai")
    assert (bedrock.resolved_llm_provider, bedrock.resolved_embedding_provider) == ("bedrock", "bedrock")
    assert openai.embedding_dimensions == 1536 and bedrock.embedding_dimensions == 1024  # vector index follows
    assert bedrock.public_dict()["llm_model"] == "anthropic.claude-opus-5-5"
    assert "bedrock_api_key" not in str(bedrock.public_dict()) and "sk-x" not in str(bedrock.public_dict())
    # Explicit pins still win (e.g. Bedrock LLM with OpenAI embeddings).
    mixed = Settings(**{**base, "openai_api_key": SecretStr("sk-x"), "use_bedrock": True, "embedding_provider": "openai"})
    assert (mixed.resolved_llm_provider, mixed.resolved_embedding_provider) == ("bedrock", "openai")
    assert isinstance(build_embedder(openai), OpenAIEmbedder) and isinstance(build_embedder(offline), HashingEmbedder)


def test_bedrock_embeddings(bedrock_settings, stub) -> None:
    embedder = build_embedder(bedrock_settings)
    assert isinstance(embedder, BedrockEmbedder) and embedder.dimensions == 512
    vectors = embedder.embed_documents(["Kafka streams events", "Neo4j stores graphs"])
    assert len(vectors) == 2 and all(len(v) == 512 for v in vectors)
    call = _requests(stub, "bedrock_embed")[-1]
    assert call["model"] == "amazon.titan-embed-text-v2:0" and call["dimensions"] == 512
    assert call["authorization"] == "Bearer" and call["bearer"] == BEDROCK_KEY  # Bedrock API key auth


def test_bedrock_structured_output_and_extraction(bedrock_settings, stub) -> None:
    llm = build_llm_client(bedrock_settings)
    assert llm is not None and llm.provider == "bedrock"
    extractor = build_graph_extractor(bedrock_settings, llm)
    assert isinstance(extractor, LLMGraphExtractor)
    result = extractor.extract("Rahul manages Project Alpha.", "c1")
    assert [(r.source, r.relationship.value, r.target) for r in result.relationships] == [("Rahul", "MANAGES", "Project Alpha")]
    call = _requests(stub, "bedrock_chat")[-1]
    assert call["model"] == "anthropic.claude-opus-5-5" and call["schema"] == "_RawExtraction"
    assert not call["tools"]  # native structured output, no forced tool call (rejected by Claude 5.x)
    assert "temperature" not in call["inference"] and call["inference"]["maxTokens"] == 16000


async def test_agent_runs_on_bedrock(bedrock_settings, sample_graph, stub) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from app.core.container import build_container

    container = build_container(bedrock_settings, None, None, checkpointer=InMemorySaver(), reader=sample_graph,
                                embedder=HashingEmbedder(256))  # the sample corpus was indexed with hashing vectors
    usage = start_usage_tracking()
    conversation = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id(TENANT_A, conversation), "tenant_id": TENANT_A,
                               "denied_document_ids": []}, "recursion_limit": RECURSION_LIMIT}
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
    assert "".join(tokens).strip() == state["answer"]  # streamed through ConverseStream
    assert state["verification"]["method"] == "llm+heuristic" and state["verification"]["passed"]
    assert {"QueryAnalysis", "LLMVerification"} <= {r["schema"] for r in _requests(stub, "bedrock_chat")}
    assert _requests(stub, "bedrock_stream")
    assert usage.total_tokens > 0 and usage.calls >= 3


def test_message_text_ignores_reasoning_blocks() -> None:
    class _Msg:
        content = [{"type": "reasoning_content", "reasoning_content": {"text": "secret"}}, {"type": "text", "text": "Hi"}]

    assert message_text(_Msg()) == "Hi" and message_text("plain") == "plain"


async def test_bedrock_failures_are_typed(bedrock_settings) -> None:
    from pydantic import BaseModel

    from app.core.errors import LLMUnavailable

    class _Schema(BaseModel):
        value: str

    broken = Settings(**{**bedrock_settings.model_dump(), "bedrock_endpoint_url": "http://127.0.0.1:9",
                         "llm_max_retries": 0, "llm_timeout_seconds": 3})
    with pytest.raises(LLMUnavailable):
        await LLMClient(broken).astructured(_Schema, [], task="t")
