"""Google Drive connector end-to-end against a fake Google API (real Postgres/Neo4j/Redis, eager Celery)."""

from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path

import pytest
from google_stub import FakeGoogle

API = "/api/v1"
SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture
def google():
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    from app.core.config import get_settings

    settings = get_settings()
    previous = settings.google_drive_api_url
    with FakeGoogle() as fake:
        settings.google_drive_api_url = f"{fake.base}/drive/v3"
        yield fake
    settings.google_drive_api_url = previous


def _admin(client) -> dict:
    r = client.post(f"{API}/auth/register", json={"email": f"drive-{uuid.uuid4().hex[:8]}@example.com",
                                                  "password": "Str0ngPassw0rd", "tenant_name": "Drive Corp"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _google_doc(text: str) -> bytes:
    import docx

    document = docx.Document()
    for para in text.split("\n"):
        document.add_paragraph(para)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_google_drive_sync_lifecycle(client, google) -> None:
    google.add("rootfolder1", "Knowledge", "application/vnd.google-apps.folder", parent="nonefolder")
    google.add("subfolder1", "Teams", "application/vnd.google-apps.folder", parent="rootfolder1")
    google.add("f-team", "team-directory.md", "text/markdown", "rootfolder1", (SAMPLES / "team-directory.md").read_bytes(),
               "2026-01-01T00:00:00.000Z")
    google.add("f-doc", "Project Delta", "application/vnd.google-apps.document", "subfolder1",
               _google_doc("Zara manages Project Delta. Project Delta uses Kafka."), "2026-01-01T00:00:00.000Z")
    google.add("f-img", "logo.png", "image/png", "rootfolder1", b"\x89PNG")
    admin = _admin(client)
    body = {"name": "Team drive", "folder_id": "rootfolder1", "service_account_json": google.service_account_json(),
            "access_groups": []}
    created = client.post(f"{API}/connectors", headers=admin, json=body)
    assert created.status_code == 201, created.text
    connector = created.json()
    assert "service_account_json" not in json.dumps(connector) and "PRIVATE KEY" not in json.dumps(connector)
    state = client.get(f"{API}/connectors/{connector['id']}", headers=admin).json()
    assert state["status"] == "IDLE", state
    assert state["last_sync_stats"]["created"] == 2 and state["last_sync_stats"]["skipped"] == ["logo.png"]
    docs = {d["filename"]: d for d in client.get(f"{API}/documents", headers=admin).json()["items"]}
    assert set(docs) == {"team-directory.md", "Project Delta.docx"}
    assert all(d["source"] == "google_drive" and d["status"] == "COMPLETED" for d in docs.values())
    answer = client.post(f"{API}/chat", headers=admin, json={"message": "Who manages Project Delta?"}).json()
    assert answer["answer"].startswith("Zara")

    # Incremental sync: unchanged files are skipped, modified files re-ingested, deleted files removed.
    google.files["f-doc"].update(content=_google_doc("Omar manages Project Delta."), modifiedTime="2026-02-01T00:00:00.000Z")
    del google.files["f-team"]
    client.post(f"{API}/connectors/{connector['id']}/sync", headers=admin)
    stats = client.get(f"{API}/connectors/{connector['id']}", headers=admin).json()["last_sync_stats"]
    assert (stats["updated"], stats["removed"], stats["unchanged"], stats["created"]) == (1, 1, 0, 0)
    filenames = [d["filename"] for d in client.get(f"{API}/documents", headers=admin).json()["items"]]
    assert filenames == ["Project Delta.docx"]
    answer = client.post(f"{API}/chat", headers=admin, json={"message": "Who manages Project Delta?"}).json()
    assert answer["answer"].startswith("Omar")
    again = client.post(f"{API}/connectors/{connector['id']}/sync", headers=admin)
    assert again.status_code == 202
    assert client.get(f"{API}/connectors/{connector['id']}", headers=admin).json()["last_sync_stats"]["unchanged"] == 1

    assert client.delete(f"{API}/connectors/{connector['id']}", headers=admin).status_code == 204
    assert client.get(f"{API}/documents", headers=admin).json()["total"] == 0


def test_connector_validation(client, google) -> None:
    admin = _admin(client)
    bad = client.post(f"{API}/connectors", headers=admin, json={"name": "x", "folder_id": "root12",
                                                                "service_account_json": '{"type": "authorized_user", "client_id": "abc"}'})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "INVALID_CREDENTIALS"
    missing = client.post(f"{API}/connectors", headers=admin, json={
        "name": "x", "folder_id": "nope123", "service_account_json": google.service_account_json()})
    assert missing.status_code == 502 and missing.json()["error"]["code"] == "GOOGLE_DRIVE_NOT_FOUND"
    member_email = f"m-{uuid.uuid4().hex[:8]}@example.com"
    client.post(f"{API}/auth/users", headers=admin, json={"email": member_email, "password": "Str0ngPassw0rd"})
    member = client.post(f"{API}/auth/login", json={"email": member_email, "password": "Str0ngPassw0rd"}).json()
    assert client.get(f"{API}/connectors", headers={"Authorization": f"Bearer {member['access_token']}"}).status_code == 403
