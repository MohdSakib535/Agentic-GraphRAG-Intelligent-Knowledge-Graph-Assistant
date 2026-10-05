"""A tiny OpenAI-compatible (and Amazon Bedrock-compatible) HTTP server used only by tests.

It lets the real ``langchain-openai`` clients (``ChatOpenAI`` with JSON-schema
structured output and token streaming, and ``OpenAIEmbeddings``) run end-to-end
without network access or an API key. Responses are derived deterministically
from the request so the agent's LLM code paths can be asserted on.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import struct
import threading
import time
import zlib
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.agents.router import candidate_entities, choose_strategy
from app.retrieval.query_parsing import expected_answer_type, relation_hints
from app.retrieval.types import LinkedEntity


def _embedding(text: str, dims: int) -> list[float]:
    vec = [0.0] * dims
    for token in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dims] += 1.0 if (h >> 8) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _question(prompt: str) -> str:
    match = re.search(r"Question:\s*(.+)", prompt)
    return match.group(1).strip() if match else prompt.strip()


def _structured(schema_name: str, prompt: str) -> dict[str, Any]:
    q = _question(prompt)
    if schema_name == "QueryAnalysis":
        known = re.search(r"Entities found in the knowledge graph for this question: (.+)", prompt)
        linked = [LinkedEntity(id=n, name=n, type=t) for n, t in re.findall(r"([^,(]+?) \((\w+)\)", known.group(1))] if known else []
        strategy, intent, reasoning = choose_strategy(q, relation_hints(q), linked, expected_answer_type(q))
        return {"standalone_question": q, "intent": intent, "entities": candidate_entities(q),
                "relationships": relation_hints(q), "temporal_constraints": [], "answer_type": expected_answer_type(q),
                "retrieval_strategy": strategy,
                "reasoning": f"stub: {reasoning}"}
    if schema_name == "ContextGrade":
        return {"relevance": 0.9, "sufficient": True, "missing_information": ""}
    if schema_name == "RewriteDecision":
        return {"rewritten_query": q, "retrieval_strategy": "HYBRID", "reasoning": "stub"}
    if schema_name == "LLMVerification":
        return {"all_claims_supported": True, "unsupported_claims": [], "introduces_outside_information": False,
                "answers_the_question": True, "confidence": 0.9}
    if schema_name == "_GeneratedCypher":
        return {"cypher": "MATCH (p:Person)-[:MANAGES]->(x:Project) RETURN p.name AS person, x.name AS project",
                "explanation": "stub"}
    if schema_name == "_SQLPlan":
        # First attempt is deliberately unsafe so the planner's validation + self-repair path is exercised.
        if "previous SQL failed" not in prompt:
            return {"sql": "SELECT * FROM read_csv('/etc/passwd')", "explanation": "bad"}
        return {"sql": 'SELECT "department", avg("annual_salary") AS avg_salary FROM data GROUP BY 1 ORDER BY 2 DESC',
                "explanation": "Average salary per department"}
    if schema_name == "_Answer":
        rows = re.search(r"Rows \(\d+\+?(?:, truncated)?\):\n(.+?)(?:\n|$)", prompt)
        top = rows.group(1).split(" | ") if rows else ["?", "?"]
        return {"answer": f"{top[0]} has the highest average salary ({top[1]})."}
    if schema_name == "JudgeVerdict":
        return {"correctness": 0.8, "faithfulness": 0.9, "reasoning": "stub judge"}
    if schema_name == "_SameEntity":
        return {"same_entity": False, "reason": "stub"}
    if schema_name == "_RawExtraction":
        text = prompt.split('"""')[1] if '"""' in prompt else prompt
        ents, rels = [], []
        for m in re.finditer(r"([A-Z][a-z]+) manages (Project [A-Z][a-z]+)", text):
            ents += [{"name": m.group(1), "type": "Person"}, {"name": m.group(2), "type": "Project"}]
            rels.append({"source": m.group(1), "relationship": "MANAGES", "target": m.group(2), "evidence": m.group(0)})
        ents.append({"name": "Hallucinated", "type": "Starship"})  # must be dropped by validation
        return {"entities": ents, "relationships": rels}
    raise ValueError(f"unknown schema {schema_name}")


def _grounded_answer(prompt: str) -> str:
    facts = re.findall(r"^- (.+?) \[(\d+)\]$", prompt, flags=re.M)
    if facts:
        return " ".join(f"{text}. [{idx}]" for text, idx in facts[:3])
    passage = re.search(r"^\[(\d+)\] \([^)]*\) (.+?[.!?])", prompt, flags=re.M)
    return f"{passage.group(2)} [{passage.group(1)}]" if passage else "INSUFFICIENT_EVIDENCE"


def _event(event_type: str, payload: dict[str, Any]) -> bytes:
    """One AWS event-stream message (the binary framing Bedrock ConverseStream uses)."""
    headers = b""
    for name, value in ((":event-type", event_type), (":content-type", "application/json"), (":message-type", "event")):
        raw = value.encode()
        headers += bytes([len(name)]) + name.encode() + bytes([7]) + struct.pack(">H", len(raw)) + raw
    body = json.dumps(payload).encode()
    total = 12 + len(headers) + len(body) + 4
    prelude = struct.pack(">II", total, len(headers))
    message = prelude + struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF) + headers + body
    return message + struct.pack(">I", zlib.crc32(message) & 0xFFFFFFFF)


def _bedrock_prompt(body: dict[str, Any]) -> str:
    parts = [b.get("text", "") for b in body.get("system") or []]
    for message in body.get("messages") or []:
        parts += [b.get("text", "") for b in message.get("content") or [] if isinstance(b, dict)]
    return "\n".join(parts)


def _bedrock_schema(body: dict[str, Any]) -> str | None:
    fmt = ((body.get("outputConfig") or {}).get("textFormat") or {})
    return ((fmt.get("structure") or {}).get("jsonSchema") or {}).get("name")


def create_stub_app(dims: int) -> FastAPI:
    app = FastAPI()
    app.state.requests = []

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        body = await request.json()
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        inputs = [i if isinstance(i, str) else " ".join(map(str, i)) for i in inputs]
        app.state.requests.append({"kind": "embeddings", "n": len(inputs), "dimensions": body.get("dimensions")})
        return JSONResponse({"object": "list", "model": body["model"],
                             "data": [{"object": "embedding", "index": i, "embedding": _embedding(t, dims)}
                                      for i, t in enumerate(inputs)],
                             "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        prompt = "\n".join(str(m.get("content", "")) for m in body["messages"])
        fmt = body.get("response_format") or {}
        schema_name = (fmt.get("json_schema") or {}).get("name")
        content = json.dumps(_structured(schema_name, prompt)) if schema_name else _grounded_answer(prompt)
        app.state.requests.append({"kind": "chat", "schema": schema_name, "stream": bool(body.get("stream"))})
        usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(content) // 4,
                 "total_tokens": len(prompt) // 4 + len(content) // 4}
        base = {"id": "chatcmpl-stub", "created": int(time.time()), "model": body["model"]}
        if body.get("stream"):
            def events() -> Any:
                for token in re.findall(r"\S+\s*", content):
                    chunk = {**base, "object": "chat.completion.chunk",
                             "choices": [{"index": 0, "delta": {"role": "assistant", "content": token}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {**base, "object": "chat.completion.chunk",
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}
                yield f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        return JSONResponse({**base, "object": "chat.completion", "usage": usage,
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": content}}]})

    # ------------------------------------------------------------ Amazon Bedrock runtime
    def _bedrock_record(request: Request, kind: str, model_id: str, body: dict[str, Any]) -> None:
        app.state.requests.append({"kind": kind, "model": model_id, "schema": _bedrock_schema(body),
                                   "authorization": request.headers.get("authorization", "").split(" ")[0],
                                   "bearer": request.headers.get("authorization", "")[7:],
                                   "inference": body.get("inferenceConfig") or {}, "tools": "toolConfig" in body})

    @app.post("/model/{model_id}/converse")
    async def converse(model_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        _bedrock_record(request, "bedrock_chat", model_id, body)
        prompt, schema = _bedrock_prompt(body), _bedrock_schema(body)
        text = json.dumps(_structured(schema, prompt)) if schema else _grounded_answer(prompt)
        usage = {"inputTokens": len(prompt) // 4, "outputTokens": len(text) // 4,
                 "totalTokens": len(prompt) // 4 + len(text) // 4}
        # A reasoning block first, as Claude 5.x returns - clients must read only the text block.
        content = [{"reasoningContent": {"reasoningText": {"text": "", "signature": "sig"}}}, {"text": text}]
        return JSONResponse({"output": {"message": {"role": "assistant", "content": content}},
                             "stopReason": "end_turn", "usage": usage, "metrics": {"latencyMs": 1}})

    @app.post("/model/{model_id}/converse-stream")
    async def converse_stream(model_id: str, request: Request) -> Response:
        body = await request.json()
        _bedrock_record(request, "bedrock_stream", model_id, body)
        prompt = _bedrock_prompt(body)
        text = _grounded_answer(prompt)
        frames = [_event("messageStart", {"role": "assistant"})]
        frames += [_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": token}})
                   for token in re.findall(r"\S+\s*", text)]
        frames += [_event("contentBlockStop", {"contentBlockIndex": 0}),
                   _event("messageStop", {"stopReason": "end_turn"}),
                   _event("metadata", {"usage": {"inputTokens": len(prompt) // 4, "outputTokens": len(text) // 4,
                                                 "totalTokens": (len(prompt) + len(text)) // 4},
                                       "metrics": {"latencyMs": 1}})]
        return Response(b"".join(frames), media_type="application/vnd.amazon.eventstream")

    @app.post("/model/{model_id}/invoke")
    async def invoke(model_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        _bedrock_record(request, "bedrock_embed", model_id, body)
        app.state.requests[-1]["dimensions"] = body.get("dimensions")
        return JSONResponse({"embedding": _embedding(body["inputText"], body.get("dimensions") or 1024),
                             "inputTextTokenCount": 1})

    return app


class StubServer:
    def __init__(self, dims: int) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.app = create_stub_app(dims)
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def bedrock_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> StubServer:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
