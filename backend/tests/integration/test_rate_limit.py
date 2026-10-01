from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def test_auth_rate_limit_returns_429(client) -> None:
    from app.core.config import get_settings
    from app.main import create_app

    settings = get_settings().model_copy(update={"rate_limit_enabled": True, "rate_limit_auth": "3/minute"})
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    body = {"email": f"rl-{uuid.uuid4().hex[:8]}@example.com", "password": "Wrong1234"}
    client_ip = f"10.{uuid.uuid4().int % 250}.{uuid.uuid4().int % 250}.{uuid.uuid4().int % 250}"
    with TestClient(app, client=(client_ip, 50000)) as c:
        codes = [c.post("/api/v1/auth/login", json=body).status_code for _ in range(4)]
        assert codes[:3] == [401, 401, 401]
        last = c.post("/api/v1/auth/login", json=body)
        assert last.status_code == 429 and last.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert int(last.headers["Retry-After"]) >= 1
