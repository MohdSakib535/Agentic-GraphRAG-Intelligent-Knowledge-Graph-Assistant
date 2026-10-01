from __future__ import annotations

import time
import uuid

import jwt
import pytest

from app.core.config import Settings, parse_rate
from app.core.errors import AuthenticationError
from app.core.logging import redact
from app.core.security import create_token, decode_token, hash_password, verify_password

SETTINGS = Settings(environment="test", jwt_secret_key="unit-test-secret-key-with-enough-length-1234")
UID, TID = str(uuid.uuid4()), str(uuid.uuid4())


def test_password_hashing_never_stores_plaintext() -> None:
    hashed = hash_password("Str0ngPassw0rd")
    assert "Str0ngPassw0rd" not in hashed and hashed.startswith("$argon2")
    assert verify_password("Str0ngPassw0rd", hashed)
    assert not verify_password("wrong", hashed)
    assert not verify_password("anything", None)  # unknown user path still runs a hash
    assert not verify_password("anything", "not-a-hash")


def test_access_and_refresh_tokens_roundtrip() -> None:
    access, claims = create_token(user_id=UID, tenant_id=TID, role="admin", token_type="access", settings=SETTINGS)
    decoded = decode_token(access, "access", SETTINGS)
    assert decoded.subject == UID and decoded.tenant_id == TID and decoded.jti == claims.jti
    refresh, _ = create_token(user_id=UID, tenant_id=TID, role="admin", token_type="refresh", settings=SETTINGS)
    with pytest.raises(AuthenticationError):
        decode_token(refresh, "access", SETTINGS)  # token type confusion is rejected


def test_tampered_and_expired_tokens_are_rejected() -> None:
    token, _ = create_token(user_id=UID, tenant_id=TID, role="member", token_type="access", settings=SETTINGS)
    with pytest.raises(AuthenticationError):
        decode_token(token[:-2] + ("A" if token[-1] != "A" else "B") + token[-1], "access", SETTINGS)
    other = Settings(environment="test", jwt_secret_key="another-secret-key-with-enough-length-9876")
    with pytest.raises(AuthenticationError):
        decode_token(token, "access", other)
    expired = jwt.encode({"sub": UID, "tid": TID, "type": "access", "jti": "x", "iat": int(time.time()) - 100,
                          "exp": int(time.time()) - 10, "iss": SETTINGS.jwt_issuer},
                         SETTINGS.jwt_secret_key.get_secret_value(), algorithm="HS256")
    with pytest.raises(AuthenticationError) as exc:
        decode_token(expired, "access", SETTINGS)
    assert exc.value.code == "TOKEN_EXPIRED"
    none_alg = jwt.encode({"sub": UID, "tid": TID, "type": "access", "jti": "x", "iat": int(time.time()),
                           "exp": int(time.time()) + 60, "iss": SETTINGS.jwt_issuer}, key=None, algorithm="none")
    with pytest.raises(AuthenticationError):
        decode_token(none_alg, "access", SETTINGS)


def test_rate_parsing_and_production_validation() -> None:
    assert parse_rate("30/minute") == (30, 60)
    assert parse_rate("10/hour") == (10, 3600)
    with pytest.raises(ValueError):
        parse_rate("lots")
    with pytest.raises(ValueError):
        Settings(environment="production", jwt_secret_key="short")
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_public_settings_never_expose_secrets() -> None:
    s = Settings(openai_api_key="sk-test-1234567890abcdef", postgres_password="pw-secret", neo4j_password="neo-secret")
    dumped = str(s.public_dict())
    assert "sk-test" not in dumped and "pw-secret" not in dumped and "neo-secret" not in dumped
    assert s.resolved_llm_provider == "openai"


def test_log_redaction() -> None:
    data = redact({"password": "p", "authorization": "Bearer abc.def.ghi", "token_usage": 42,
                   "msg": "key sk-abcdefghijklmnop and eyJhbGciOi.eyJzdWIiOi.c2lnbmF0dXJl"})
    assert data["password"] == "[REDACTED]" and data["authorization"] == "[REDACTED]"
    assert data["token_usage"] == 42
    assert "sk-abcdef" not in data["msg"] and "eyJhbGciOi" not in data["msg"]
