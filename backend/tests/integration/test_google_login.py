"""Sign in with Google (OIDC + PKCE) against a fake Google: provisioning, linking, domains and replay safety."""

from __future__ import annotations

import os
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from google_stub import FakeGoogle
from pydantic import SecretStr

API = "/api/v1"


@pytest.fixture
def google():
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    from app.core.config import get_settings

    settings = get_settings()
    fields = ("google_client_id", "google_client_secret", "google_token_url", "google_jwks_url", "google_allowed_domains")
    previous = {f: getattr(settings, f) for f in fields}
    with FakeGoogle() as fake:
        settings.google_client_id = "client-123.apps.googleusercontent.com"
        settings.google_client_secret = SecretStr("shh")
        settings.google_token_url = f"{fake.base}/token"
        settings.google_jwks_url = f"{fake.base}/oauth2/v3/certs"
        settings.google_allowed_domains = []
        yield fake
    for f, v in previous.items():
        setattr(settings, f, v)


def _sign_in(client, google, **identity) -> dict:
    start = client.get(f"{API}/auth/google/login", follow_redirects=False)
    assert start.status_code == 302, start.text
    code, state = google.authorize(start.headers["location"], **identity)
    back = client.get(f"{API}/auth/google/callback", params={"code": code, "state": state}, follow_redirects=False)
    assert back.status_code == 302
    query = {k: v[0] for k, v in parse_qs(urlparse(back.headers["location"]).query).items()}
    assert "access_token" not in back.headers["location"]
    if "sso_error" in query:
        return {"error": query["sso_error"]}
    out = client.post(f"{API}/auth/google/exchange", json={"code": query["sso_code"]})
    assert out.status_code == 200, out.text
    replay = client.post(f"{API}/auth/google/exchange", json={"code": query["sso_code"]})
    assert replay.status_code == 401 and replay.json()["error"]["code"] == "INVALID_SSO_CODE"
    return out.json()


def test_google_login_provisioning(client, google) -> None:
    domain = f"acme-{uuid.uuid4().hex[:6]}.com"
    founder = _sign_in(client, google, sub=f"g-{uuid.uuid4().hex}", email=f"ceo@{domain}", hd=domain)
    assert founder["user"]["role"] == "admin" and founder["user"]["auth_provider"] == "google"
    colleague = _sign_in(client, google, sub=f"g-{uuid.uuid4().hex}", email=f"dev@{domain}", hd=domain)
    assert colleague["user"]["role"] == "member"
    assert colleague["user"]["tenant_id"] == founder["user"]["tenant_id"]  # Workspace domain joins the tenant
    me = client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {colleague['access_token']}"}).json()
    assert me["tenant"]["id"] == founder["user"]["tenant_id"]
    # SSO-only accounts cannot use password login.
    bad = client.post(f"{API}/auth/login", json={"email": f"dev@{domain}", "password": "anything-goes-123"})
    assert bad.status_code == 401
    # Personal Gmail accounts get their own tenant.
    solo = _sign_in(client, google, sub=f"g-{uuid.uuid4().hex}", email=f"x{uuid.uuid4().hex[:6]}@gmail.com")
    assert solo["user"]["tenant_id"] != founder["user"]["tenant_id"] and solo["user"]["role"] == "admin"


def test_google_login_links_existing_password_account(client, google) -> None:
    email = f"linked-{uuid.uuid4().hex[:8]}@example.com"
    reg = client.post(f"{API}/auth/register", json={"email": email, "password": "Str0ngPassw0rd", "tenant_name": "Pw"})
    sub = f"g-{uuid.uuid4().hex}"
    first = _sign_in(client, google, sub=sub, email=email)
    assert first["user"]["id"] == reg.json()["user"]["id"]
    again = _sign_in(client, google, sub=sub, email=email.upper())
    assert again["user"]["id"] == reg.json()["user"]["id"]


def test_google_login_rejections(client, google) -> None:
    assert _sign_in(client, google, sub="g-unverified", email="u@example.com", email_verified=False) == \
        {"error": "GOOGLE_EMAIL_UNVERIFIED"}
    assert _sign_in(client, google, sub="g-nonce", email="n@example.com", nonce="forged") == \
        {"error": "GOOGLE_TOKEN_INVALID"}
    assert _sign_in(client, google, sub="g-aud", email="a@example.com", client_id="someone-else") == \
        {"error": "GOOGLE_TOKEN_INVALID"}
    from app.core.config import get_settings

    get_settings().google_allowed_domains = ["corp.example"]
    assert _sign_in(client, google, sub="g-dom", email="d@other.example") == {"error": "GOOGLE_DOMAIN_NOT_ALLOWED"}
    # Forged / replayed state.
    back = client.get(f"{API}/auth/google/callback", params={"code": "x", "state": "forged"}, follow_redirects=False)
    assert "sso_error=GOOGLE_STATE_INVALID" in back.headers["location"]


def test_google_login_disabled(client) -> None:
    r = client.get(f"{API}/auth/google/login", follow_redirects=False)
    assert r.status_code == 404 and r.json()["error"]["code"] == "GOOGLE_LOGIN_DISABLED"
