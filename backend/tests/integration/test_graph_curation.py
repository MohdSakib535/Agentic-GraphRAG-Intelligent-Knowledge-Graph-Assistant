"""Admin graph curation against real Neo4j: edits are validated, audited, ACL-safe and survive re-ingestion."""

from __future__ import annotations

import uuid
from pathlib import Path

API = "/api/v1"
SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
PASSWORD = "Str0ngPassw0rd"


def _register(client) -> dict:
    r = client.post(f"{API}/auth/register", json={"email": f"curator-{uuid.uuid4().hex[:8]}@example.com",
                                                  "password": PASSWORD, "tenant_name": "Curate Corp"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _entity(client, h: dict, name: str) -> dict | None:
    rows = client.get(f"{API}/graph/entities", headers=h, params={"q": name}).json()
    return next((e for e in rows if e["name"].lower() == name.lower()), None)


def test_graph_curation_lifecycle(client) -> None:
    admin = _register(client)
    up = client.post(f"{API}/documents/upload", headers=admin,
                     files={"file": ("team.md", (SAMPLES / "team-directory.md").read_bytes())})
    assert up.status_code == 202, up.text
    amit, priya = _entity(client, admin, "Amit"), _entity(client, admin, "Priya")
    kafka, redis_ = _entity(client, admin, "Kafka"), _entity(client, admin, "Redis")
    assert amit and priya and kafka and redis_

    # Members cannot curate.
    member_email = f"member-{uuid.uuid4().hex[:8]}@example.com"
    client.post(f"{API}/auth/users", headers=admin, json={"email": member_email, "password": PASSWORD})
    member_token = client.post(f"{API}/auth/login", json={"email": member_email, "password": PASSWORD}).json()
    member = {"Authorization": f"Bearer {member_token['access_token']}"}
    assert client.patch(f"{API}/graph/entities/{amit['id']}", headers=member, json={"name": "X"}).status_code == 403

    # Rename + describe.
    r = client.patch(f"{API}/graph/entities/{amit['id']}", headers=admin,
                     json={"name": "Amit Kumar", "description": "Senior backend developer"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Amit Kumar" and "Amit" in r.json()["aliases"]
    clash = client.patch(f"{API}/graph/entities/{priya['id']}", headers=admin, json={"name": "amit kumar"})
    assert clash.status_code == 409 and clash.json()["error"]["code"] == "ENTITY_EXISTS"
    bad_type = client.patch(f"{API}/graph/entities/{priya['id']}", headers=admin, json={"type": "Spaceship"})
    assert bad_type.status_code == 422

    # Merge Redis into Kafka: relationships and aliases move, the duplicate disappears.
    merged = client.post(f"{API}/graph/entities/merge", headers=admin,
                         json={"keep_id": kafka["id"], "merge_ids": [redis_["id"]]})
    assert merged.status_code == 200, merged.text
    assert "Redis" in merged.json()["aliases"]
    assert client.get(f"{API}/graph/entities/{redis_['id']}", headers=admin).status_code == 404
    detail = client.get(f"{API}/graph/entities/{kafka['id']}", headers=admin).json()
    assert any(f["source"] == "Amit Kumar" and f["relationship"] == "USES" for f in detail["relationships"])

    # Relationships are validated against the schema; manual ones are flagged.
    invalid = client.post(f"{API}/graph/relationships", headers=admin,
                          json={"source_id": amit["id"], "type": "MANAGES", "target_id": kafka["id"]})
    assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "INVALID_RELATIONSHIP"
    structural = client.post(f"{API}/graph/relationships", headers=admin,
                             json={"source_id": amit["id"], "type": "MENTIONS", "target_id": priya["id"]})
    assert structural.status_code == 422
    added = client.post(f"{API}/graph/relationships", headers=admin,
                        json={"source_id": amit["id"], "type": "REPORTS_TO", "target_id": priya["id"],
                              "evidence": "Org chart 2026"})
    assert added.status_code == 201 and added.json()["manual"] is True

    # Re-ingestion keeps the curation: name, merge and manual relationship all survive.
    doc_id = up.json()["document"]["id"]
    assert client.post(f"{API}/documents/{doc_id}/reprocess", headers=admin).status_code == 202
    assert _entity(client, admin, "Redis") is None
    renamed = client.get(f"{API}/graph/entities/{amit['id']}", headers=admin).json()
    assert renamed["entity"]["name"] == "Amit Kumar"
    facts = {(f["relationship"], f["target"], f["manual"]) for f in renamed["relationships"] if f["source"] == "Amit Kumar"}
    assert ("REPORTS_TO", "Priya", True) in facts and ("REPORTS_TO", "Rahul", False) in facts

    # Delete a relationship and an entity.
    ref = {"source_id": amit["id"], "type": "REPORTS_TO", "target_id": priya["id"]}
    assert client.post(f"{API}/graph/relationships/delete", headers=admin, json=ref).status_code == 204
    assert client.post(f"{API}/graph/relationships/delete", headers=admin, json=ref).status_code == 404
    assert client.delete(f"{API}/graph/entities/{priya['id']}", headers=admin).status_code == 204
    assert _entity(client, admin, "Priya") is None
    # Chat sees the curated graph immediately (answer cache invalidated).
    answer = client.post(f"{API}/chat", headers=admin, json={"message": "Who reports to Rahul?"}).json()["answer"]
    assert "Amit Kumar" in answer and "Priya" not in answer
