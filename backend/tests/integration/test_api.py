"""End-to-end API tests: auth, documents/ingestion, graph, search, chat, SSE, isolation, evaluation."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
API = "/api/v1"
PASSWORD = "Str0ngPassw0rd"


def register(client, tenant: str = "TestCorp") -> dict:
    email = f"user-{uuid.uuid4().hex[:10]}@example.com"
    r = client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD, "tenant_name": tenant})
    assert r.status_code == 201, r.text
    return {**r.json(), "email": email}


def auth(user: dict) -> dict:
    return {"Authorization": f"Bearer {user['access_token']}"}


@pytest.fixture(scope="module")
def tenant_a(client) -> dict:
    user = register(client, "Tenant A")
    for name in ("architecture.pdf", "project-overview.docx", "team-directory.md", "technology-glossary.txt"):
        r = client.post(f"{API}/documents/upload", headers=auth(user), files={"file": (name, (SAMPLES / name).read_bytes())})
        assert r.status_code == 202, r.text
    return user


@pytest.fixture(scope="module")
def tenant_b(client) -> dict:
    return register(client, "Tenant B")


def chat(client, user: dict, message: str, conversation_id: str | None = None) -> dict:
    body = {"message": message, **({"conversation_id": conversation_id} if conversation_id else {})}
    r = client.post(f"{API}/chat", headers=auth(user), json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ health
def test_health(client) -> None:
    assert client.get(f"{API}/health").json() == {"status": "ok"}
    ready = client.get(f"{API}/health/ready").json()
    assert ready["status"] == "ok" and set(ready["checks"]) == {"postgres", "neo4j", "redis"}
    assert client.get("/docs").status_code == 200
    assert "/api/v1/chat/stream" in client.get("/openapi.json").json()["paths"]


# -------------------------------------------------------------------- auth
def test_auth_flow(client) -> None:
    user = register(client)
    r = client.post(f"{API}/auth/register", json={"email": user["email"], "password": PASSWORD, "tenant_name": "XY"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "EMAIL_ALREADY_REGISTERED"
    weak = client.post(f"{API}/auth/register", json={"email": "w@example.com", "password": "weakpass", "tenant_name": "XY"})
    assert weak.status_code == 422 and weak.json()["success"] is False
    bad = client.post(f"{API}/auth/login", json={"email": user["email"], "password": "Wrong1234"})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "INVALID_CREDENTIALS"
    login = client.post(f"{API}/auth/login", json={"email": user["email"], "password": PASSWORD}).json()
    me = client.get(f"{API}/auth/me", headers=auth(login)).json()
    assert me["user"]["email"] == user["email"] and me["user"]["role"] == "admin"
    assert "password_hash" not in json.dumps(me) and PASSWORD not in json.dumps(me)

    rotated = client.post(f"{API}/auth/refresh", json={"refresh_token": login["refresh_token"]})
    assert rotated.status_code == 200 and rotated.json()["refresh_token"] != login["refresh_token"]
    reuse = client.post(f"{API}/auth/refresh", json={"refresh_token": login["refresh_token"]})
    assert reuse.status_code == 401 and reuse.json()["error"]["code"] == "TOKEN_REVOKED"
    # Reuse detection revoked the whole family, including the newly rotated token.
    assert client.post(f"{API}/auth/refresh", json={"refresh_token": rotated.json()["refresh_token"]}).status_code == 401

    fresh = client.post(f"{API}/auth/login", json={"email": user["email"], "password": PASSWORD}).json()
    out = client.post(f"{API}/auth/logout", headers=auth(fresh), json={"refresh_token": fresh["refresh_token"]})
    assert out.status_code == 204
    revoked = client.get(f"{API}/auth/me", headers=auth(fresh))
    assert revoked.status_code == 401 and revoked.json()["error"]["code"] == "TOKEN_REVOKED"
    assert client.post(f"{API}/auth/refresh", json={"refresh_token": fresh["refresh_token"]}).status_code == 401


def test_errors_use_consistent_envelope(client) -> None:
    r = client.get(f"{API}/documents")
    body = r.json()
    assert r.status_code == 401 and body["success"] is False and body["error"]["code"] == "NOT_AUTHENTICATED"
    assert body["request_id"] and r.headers["X-Request-ID"] == body["request_id"]
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
    garbage = client.get(f"{API}/documents", headers={"Authorization": "Bearer not.a.jwt"})
    assert garbage.status_code == 401 and "Traceback" not in garbage.text


# --------------------------------------------------------------- documents
def test_document_ingestion(client, tenant_a) -> None:
    docs = client.get(f"{API}/documents", headers=auth(tenant_a)).json()
    assert docs["total"] == 4
    assert {d["status"] for d in docs["items"]} == {"COMPLETED"}
    pdf = next(d for d in docs["items"] if d["file_type"] == "pdf")
    assert pdf["page_count"] == 4 and pdf["chunk_count"] > 0 and pdf["entity_count"] > 0
    status = client.get(f"{API}/documents/{pdf['id']}/status", headers=auth(tenant_a)).json()
    stages = [s["stage"] for s in status["job"]["stage_history"]]
    assert stages == ["PARSING", "CLEANING", "CHUNKING", "EXTRACTING_ENTITIES", "EXTRACTING_RELATIONSHIPS",
                      "RESOLVING_ENTITIES", "BUILDING_GRAPH", "EMBEDDING", "INDEXING", "COMPLETED"]
    assert status["job"]["progress"] == 100
    dup = client.post(f"{API}/documents/upload", headers=auth(tenant_a),
                      files={"file": ("copy.md", (SAMPLES / "team-directory.md").read_bytes())})
    assert dup.status_code == 409 and dup.json()["error"]["code"] == "DOCUMENT_ALREADY_EXISTS"
    bad = client.post(f"{API}/documents/upload", headers=auth(tenant_a), files={"file": ("x.exe", b"MZ")})
    assert bad.status_code == 415 and bad.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
    fake = client.post(f"{API}/documents/upload", headers=auth(tenant_a), files={"file": ("x.pdf", b"hello")})
    assert fake.status_code == 400 and fake.json()["error"]["code"] == "INVALID_FILE"


def test_knowledge_graph_endpoints(client, tenant_a) -> None:
    stats = client.get(f"{API}/graph/stats", headers=auth(tenant_a)).json()
    assert stats["entities"] >= 15 and stats["relationships"] >= 30 and stats["documents"] == 4
    assert stats["entities_by_type"]["Person"] == 4
    rahul = client.get(f"{API}/graph/entities", headers=auth(tenant_a), params={"q": "rahul"}).json()[0]
    detail = client.get(f"{API}/graph/entities/{rahul['id']}", headers=auth(tenant_a)).json()
    rels = {(r["source"], r["relationship"], r["target"]) for r in detail["relationships"]}
    assert ("Rahul", "MANAGES", "Project Alpha") in rels and ("Amit", "REPORTS_TO", "Rahul") in rels
    assert detail["sources"] and detail["sources"][0]["source_filename"]
    sub = client.get(f"{API}/graph/subgraph", headers=auth(tenant_a), params={"entity_id": rahul["id"], "depth": 2}).json()
    assert len(sub["nodes"]) > 5 and all(e["source_id"] and e["target_id"] for e in sub["edges"])


@pytest.mark.parametrize("strategy", ["VECTOR", "GRAPH", "HYBRID"])
def test_search_strategies(client, tenant_a, strategy) -> None:
    r = client.post(f"{API}/search", headers=auth(tenant_a), json={"query": "Who manages Project Alpha?", "strategy": strategy})
    body = r.json()
    assert r.status_code == 200 and body["strategy"] == strategy
    if strategy == "VECTOR":
        assert body["chunks"] and not body["facts"]
    else:
        assert any(f["source"] == "Rahul" and f["target"] == "Project Alpha" for f in body["facts"])


# -------------------------------------------------------------------- chat
def test_chat_end_to_end(client, tenant_a) -> None:
    vector = chat(client, tenant_a, "What is Kafka?")
    assert vector["retrieval_strategy"] == "VECTOR" and "distributed event streaming" in vector["answer"]
    assert vector["sources"] and vector["confidence"] > 0.5
    graph = chat(client, tenant_a, "Who manages Project Alpha?")
    assert graph["retrieval_strategy"] == "GRAPH" and graph["answer"].startswith("Rahul")
    hybrid = chat(client, tenant_a, "Which developers work on Kafka projects managed by Rahul?")
    assert hybrid["retrieval_strategy"] == "HYBRID" and "Amit" in hybrid["answer"] and "Neha" in hybrid["answer"]
    assert hybrid["graph_evidence"] and hybrid["retrieved_chunks"] and hybrid["trace"]
    unsupported = chat(client, tenant_a, "What is the salary of the CEO of TechCorp?")
    assert unsupported["answer"].startswith("I don't have enough information") and unsupported["sources"] == []


def test_conversation_memory_and_history(client, tenant_a) -> None:
    first = chat(client, tenant_a, "Who manages Project Alpha?")
    follow = chat(client, tenant_a, "What technologies does that project use?", first["conversation_id"])
    assert "Kafka" in follow["answer"] and "Redis" in follow["answer"]
    messages = client.get(f"{API}/chat/conversations/{first['conversation_id']}/messages", headers=auth(tenant_a)).json()
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["sources"] and messages[1]["retrieval_strategy"] == "GRAPH"
    convs = client.get(f"{API}/chat/conversations", headers=auth(tenant_a)).json()
    assert any(c["id"] == first["conversation_id"] for c in convs)


def test_sse_streaming(client, tenant_a) -> None:
    events: list[str] = []
    completed = None
    with client.stream("POST", f"{API}/chat/stream", headers=auth(tenant_a), json={"message": "Which projects use Kafka?"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        current = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                current = line[6:].strip()
                events.append(current)
            elif line.startswith("data:") and current == "completed":
                completed = json.loads(line[5:])
    for expected in ("agent_started", "query_analyzed", "retrieval_started", "retrieval_completed", "reasoning",
                     "token", "citation", "verification", "completed"):
        assert expected in events, expected
    assert completed and "Project Alpha" in completed["answer"] and "Project Gamma" in completed["answer"]


# --------------------------------------------------------------- isolation
def test_tenant_isolation(client, tenant_a, tenant_b) -> None:
    doc_id = client.get(f"{API}/documents", headers=auth(tenant_a)).json()["items"][0]["id"]
    assert client.get(f"{API}/documents", headers=auth(tenant_b)).json()["total"] == 0
    assert client.get(f"{API}/documents/{doc_id}", headers=auth(tenant_b)).status_code == 404
    assert client.delete(f"{API}/documents/{doc_id}", headers=auth(tenant_b)).status_code == 404
    assert client.get(f"{API}/graph/stats", headers=auth(tenant_b)).json()["entities"] == 0
    assert client.get(f"{API}/graph/entities", headers=auth(tenant_b), params={"q": "Rahul"}).json() == []
    search = client.post(f"{API}/search", headers=auth(tenant_b), json={"query": "Kafka Project Alpha", "strategy": "HYBRID"}).json()
    assert search["chunks"] == [] and search["facts"] == []
    answer = chat(client, tenant_b, "Who manages Project Alpha?")
    assert answer["answer"].startswith("I don't have enough information")
    conv_a = chat(client, tenant_a, "What is Redis?")["conversation_id"]
    assert client.get(f"{API}/chat/conversations/{conv_a}/messages", headers=auth(tenant_b)).status_code == 404
    hijack = client.post(f"{API}/chat", headers=auth(tenant_b), json={"message": "hi", "conversation_id": conv_a})
    assert hijack.status_code == 404


def test_delete_document_prunes_graph(client) -> None:
    user = register(client, "Delete Corp")
    up = client.post(f"{API}/documents/upload", headers=auth(user),
                     files={"file": ("team.md", (SAMPLES / "team-directory.md").read_bytes())}).json()
    before = client.get(f"{API}/graph/stats", headers=auth(user)).json()
    assert before["entities"] > 0 and before["chunks"] > 0
    assert client.delete(f"{API}/documents/{up['document']['id']}", headers=auth(user)).status_code == 204
    after = client.get(f"{API}/graph/stats", headers=auth(user)).json()
    assert after == {**after, "entities": 0, "relationships": 0, "chunks": 0, "documents": 0}


# -------------------------------------------------------------- evaluation
def test_evaluation_run(client, tenant_a) -> None:
    run = client.post(f"{API}/evaluation/run", headers=auth(tenant_a),
                      json={"systems": ["vector_rag", "agentic_graphrag"], "categories": ["multi_hop", "unanswerable"]})
    assert run.status_code == 202
    results = client.get(f"{API}/evaluation/results", headers=auth(tenant_a)).json()
    assert results["run"]["status"] == "COMPLETED"
    summary = results["run"]["summary"]
    assert summary["agentic_graphrag"]["accuracy"] >= summary["vector_rag"]["accuracy"]
    assert summary["agentic_graphrag"]["by_category"]["unanswerable"] == 1.0
    assert len(results["results"]) == 2 * 13


def test_document_level_permissions(client) -> None:
    admin = register(client, "ACL Corp")
    member_email = f"member-{uuid.uuid4().hex[:8]}@example.com"
    created = client.post(f"{API}/auth/users", headers=auth(admin),
                          json={"email": member_email, "password": PASSWORD, "groups": ["engineering"]})
    assert created.status_code == 201, created.text
    member = client.post(f"{API}/auth/login", json={"email": member_email, "password": PASSWORD}).json()
    for name, groups in (("team-directory.md", "hr"), ("project-overview.docx", "")):
        r = client.post(f"{API}/documents/upload", headers=auth(admin), data={"access_groups": groups},
                        files={"file": (name, (SAMPLES / name).read_bytes())})
        assert r.status_code == 202, r.text
    hr_doc = next(d for d in client.get(f"{API}/documents", headers=auth(admin)).json()["items"]
                  if d["filename"] == "team-directory.md")
    assert hr_doc["access_groups"] == ["hr"]
    # Member (engineering) cannot see the HR document anywhere.
    assert [d["filename"] for d in client.get(f"{API}/documents", headers=auth(member)).json()["items"]] == \
        ["project-overview.docx"]
    assert client.get(f"{API}/documents/{hr_doc['id']}", headers=auth(member)).status_code == 404
    assert chat(client, member, "Who reports to Rahul?")["answer"].startswith("I don't have enough information")
    assert chat(client, admin, "Who reports to Rahul?")["answer"].startswith("Amit")
    # Non-admins cannot change access or restrict to groups they don't belong to.
    assert client.put(f"{API}/documents/{hr_doc['id']}/access", headers=auth(member),
                      json={"access_groups": []}).status_code == 403
    bad = client.post(f"{API}/documents/upload", headers=auth(member), data={"access_groups": "hr"},
                      files={"file": ("x.md", b"# X\n\nhello world")})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "INVALID_ACCESS_GROUPS"
    # Granting the group makes the document visible immediately (cache scoped by permissions).
    users = client.get(f"{API}/auth/users", headers=auth(admin)).json()
    member_id = next(u["id"] for u in users if u["email"] == member_email)
    assert client.patch(f"{API}/auth/users/{member_id}", headers=auth(admin),
                        json={"groups": ["engineering", "hr"]}).json()["groups"] == ["engineering", "hr"]
    assert chat(client, member, "Who reports to Rahul?")["answer"].startswith("Amit")


def test_answer_cache_respects_memory_and_permissions(client) -> None:
    admin = register(client, "Cache Corp")
    for name in ("project-overview.docx", "architecture.pdf"):
        client.post(f"{API}/documents/upload", headers=auth(admin), files={"file": (name, (SAMPLES / name).read_bytes())})
    first = chat(client, admin, "Which projects use Kafka?")
    second = chat(client, admin, "which projects use kafka")  # normalised question, new conversation
    assert first["cached"] is False and second["cached"] is True and second["answer"] == first["answer"]
    follow = chat(client, admin, "Who manages that project?", second["conversation_id"])  # memory restored
    assert follow["cached"] is False and "Rahul" in follow["answer"]
    # A user with a different permission scope never receives the cached answer.
    email = f"m-{uuid.uuid4().hex[:8]}@example.com"
    client.post(f"{API}/auth/users", headers=auth(admin), json={"email": email, "password": PASSWORD})
    docs = client.get(f"{API}/documents", headers=auth(admin)).json()["items"]
    pdf = next(d for d in docs if d["filename"] == "architecture.pdf")
    client.put(f"{API}/documents/{pdf['id']}/access", headers=auth(admin), json={"access_groups": ["arch"]})
    member = client.post(f"{API}/auth/login", json={"email": email, "password": PASSWORD}).json()
    restricted = chat(client, member, "Which projects use Kafka?")
    assert restricted["cached"] is False and "Project Alpha" in restricted["answer"]
    assert restricted["sources"] and all(s["source_filename"] != "architecture.pdf" for s in restricted["sources"])
