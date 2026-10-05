"""System test of the running Docker stack (spec section 64 end-to-end flow).

    docker compose up --build -d
    pip install pytest httpx && pytest tests/e2e          # STACK_API_URL defaults to http://localhost:8000/api/v1

Uses the real Celery worker, PostgreSQL, Neo4j and Redis. Skipped when the stack is not running.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import httpx
import pytest

API = os.getenv("STACK_API_URL", "http://localhost:8000/api/v1")
SAMPLES = Path(__file__).resolve().parents[2] / "backend" / "data" / "samples"
PASSWORD = "Str0ngPassw0rd"
INSUFFICIENT = "I don't have enough information in the uploaded knowledge base to answer this reliably."


def _stack_up() -> bool:
    try:
        return httpx.get(f"{API}/health/ready", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _stack_up(), reason=f"stack not reachable at {API}")


def register(client: httpx.Client, tenant: str) -> dict:
    email = f"e2e-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD, "tenant_name": tenant,
                                                  "full_name": "E2E User"})
    assert r.status_code == 201, r.text
    login = client.post(f"{API}/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.fixture(scope="module")
def client():
    with httpx.Client(timeout=120) as c:
        yield c


@pytest.fixture(scope="module")
def tenant(client: httpx.Client) -> dict:
    headers = register(client, "E2E TechCorp")
    ids = []
    for name in ("architecture.pdf", "project-overview.docx", "team-directory.md", "technology-glossary.txt"):
        r = client.post(f"{API}/documents/upload", headers=headers, files={"file": (name, (SAMPLES / name).read_bytes())})
        assert r.status_code == 202, r.text
        ids.append(r.json()["document"]["id"])
    deadline = time.time() + 180
    while time.time() < deadline:
        statuses = [client.get(f"{API}/documents/{i}/status", headers=headers).json()["status"] for i in ids]
        if all(s in ("COMPLETED", "FAILED") for s in statuses):
            break
        time.sleep(1)
    assert statuses == ["COMPLETED"] * 4, statuses
    return headers


def ask(client: httpx.Client, headers: dict, message: str, conversation_id: str | None = None) -> dict:
    body = {"message": message, **({"conversation_id": conversation_id} if conversation_id else {})}
    r = client.post(f"{API}/chat", headers=headers, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_ingestion_chunks_entities_relationships_embeddings(client, tenant) -> None:
    docs = client.get(f"{API}/documents", headers=tenant).json()["items"]
    assert all(d["chunk_count"] > 0 for d in docs)
    pdf = next(d for d in docs if d["filename"] == "architecture.pdf")
    assert pdf["page_count"] == 4 and pdf["entity_count"] > 5 and pdf["relationship_count"] > 5
    stats = client.get(f"{API}/graph/stats", headers=tenant).json()
    assert stats["chunks"] >= 8 and stats["entities"] >= 15 and stats["relationships"] >= 30
    assert stats["relationships_by_type"].get("MANAGES", 0) >= 3
    hits = client.post(f"{API}/search", headers=tenant, json={"query": "event streaming", "strategy": "VECTOR"}).json()
    assert hits["chunks"] and hits["chunks"][0]["score"] > 0  # embeddings + vector index work


def test_agent_strategies_citations_and_abstention(client, tenant) -> None:
    kafka = ask(client, tenant, "What is Kafka?")
    assert kafka["retrieval_strategy"] == "VECTOR" and "event streaming" in kafka["answer"]
    manages = ask(client, tenant, "Who manages Project Alpha?")
    assert manages["retrieval_strategy"] == "GRAPH" and manages["answer"].startswith("Rahul")
    hybrid = ask(client, tenant, "Which developers work on Kafka projects managed by Rahul?")
    assert hybrid["retrieval_strategy"] == "HYBRID"
    assert "Amit" in hybrid["answer"] and "Neha" in hybrid["answer"]
    for result in (kafka, manages, hybrid):
        assert result["sources"] and all(s.get("snippet") for s in result["sources"])  # citations
        assert result["verification"]["passed"]
    assert any(f["relationship"] == "MANAGES" for f in hybrid["graph_evidence"])  # graph evidence
    unsupported = ask(client, tenant, "Tell me something not contained in the documents.")
    assert unsupported["answer"] == INSUFFICIENT and unsupported["sources"] == []


def test_conversation_memory(client, tenant) -> None:
    first = ask(client, tenant, "Who manages Project Alpha?")
    follow = ask(client, tenant, "What technologies does that project use?", first["conversation_id"])
    for tech in ("Kafka", "Redis", "PostgreSQL", "FastAPI"):
        assert tech in follow["answer"]


def test_sse_streaming(client, tenant) -> None:
    events = []
    with client.stream("POST", f"{API}/chat/stream", headers=tenant, json={"message": "Which projects use Kafka?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append(line[6:].strip())
    assert {"agent_started", "query_analyzed", "retrieval_started", "retrieval_completed", "reasoning", "token",
            "citation", "verification", "completed"} <= set(events)


def test_tenant_isolation(client, tenant) -> None:
    other = register(client, "E2E Other")
    assert client.get(f"{API}/documents", headers=other).json()["total"] == 0
    assert client.get(f"{API}/graph/stats", headers=other).json()["entities"] == 0
    assert ask(client, other, "Who manages Project Alpha?")["answer"] == INSUFFICIENT
    doc = client.get(f"{API}/documents", headers=tenant).json()["items"][0]["id"]
    assert client.get(f"{API}/documents/{doc}", headers=other).status_code == 404


def test_frontend_is_served() -> None:
    url = os.getenv("STACK_UI_URL", "http://localhost:8501")
    try:
        assert httpx.get(f"{url}/_stcore/health", timeout=5).text == "ok"
    except httpx.HTTPError:
        pytest.skip("frontend not reachable")


def test_evaluation_via_worker(client, tenant) -> None:
    run = client.post(f"{API}/evaluation/run", headers=tenant, json={}).json()
    deadline = time.time() + 300
    while time.time() < deadline:
        data = client.get(f"{API}/evaluation/results", headers=tenant, params={"run_id": run["id"]}).json()
        if data["run"]["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(2)
    assert data["run"]["status"] == "COMPLETED", data["run"]
    summary = data["run"]["summary"]
    print(json.dumps({k: v for k, v in summary.items() if k != "options"}, indent=1))
    assert summary["agentic_graphrag"]["accuracy"] >= 0.9
    assert summary["agentic_graphrag"]["accuracy"] > summary["vector_rag"]["accuracy"]


def test_chat_with_csv(client) -> None:
    headers = register(client, "E2E Analytics")
    csv = (SAMPLES.parent / "datasets" / "employees.csv").read_bytes()
    up = client.post(f"{API}/datasets/upload", headers=headers, files={"file": ("employees.csv", csv, "text/csv")})
    assert up.status_code == 201, up.text
    r = client.post(f"{API}/datasets/{up.json()['id']}/query", headers=headers,
                    json={"question": "How many employees are in each department?"})
    assert r.status_code == 200, r.text
    assert r.json()["sql"].lower().startswith(("select", "with")) and r.json()["rows"]
    bad = client.post(f"{API}/datasets/{up.json()['id']}/query", headers=headers,
                      json={"question": "x", "sql": "SELECT * FROM read_csv('/etc/passwd')"})
    assert bad.status_code in (400, 422) and bad.json()["error"]["code"] == "SQL_REJECTED"


def test_document_permissions_and_curation(client, tenant) -> None:
    # Restricted upload is processed by the worker and hidden from members outside the group.
    member_email = f"e2e-member-{uuid.uuid4().hex[:6]}@example.com"
    assert client.post(f"{API}/auth/users", headers=tenant,
                       json={"email": member_email, "password": PASSWORD}).status_code == 201
    member = client.post(f"{API}/auth/login", json={"email": member_email, "password": PASSWORD}).json()
    member_h = {"Authorization": f"Bearer {member['access_token']}"}
    secret = b"# Board memo\n\nOrion Labs acquires Zephyr Systems. Zephyr Systems uses Kafka."
    up = client.post(f"{API}/documents/upload", headers=tenant, data={"access_groups": "board"},
                     files={"file": ("memo.md", secret)})
    assert up.status_code == 202, up.text
    doc_id = up.json()["document"]["id"]
    for _ in range(120):
        if client.get(f"{API}/documents/{doc_id}/status", headers=tenant).json()["status"] == "COMPLETED":
            break
        time.sleep(1)
    assert client.get(f"{API}/documents/{doc_id}", headers=member_h).status_code == 404
    hits = client.get(f"{API}/graph/entities", headers=member_h, params={"q": "Zephyr"}).json()
    assert hits == []
    # Admin curation: rename an entity and see it in search.
    zephyr = client.get(f"{API}/graph/entities", headers=tenant, params={"q": "Zephyr"}).json()
    assert zephyr, "entity extracted from the restricted memo"
    r = client.patch(f"{API}/graph/entities/{zephyr[0]['id']}", headers=tenant, json={"name": "Zephyr Systems Ltd"})
    assert r.status_code == 200, r.text
    assert client.patch(f"{API}/graph/entities/{zephyr[0]['id']}", headers=member_h,
                        json={"name": "nope"}).status_code == 403


def test_metrics_and_observability() -> None:
    base = API.rsplit("/api/", 1)[0]
    text = httpx.get(f"{base}/metrics", timeout=5).text
    assert "graphrag_http_requests_total" in text and "graphrag_ingestion_jobs" in text
    assert httpx.get(f"{API}/auth/providers", timeout=5).json()["password"] is True
