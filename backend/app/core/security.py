"""Password hashing and JWT handling."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError

TokenType = Literal["access", "refresh"]

_password_hash = PasswordHash((Argon2Hasher(),))
# Used to equalise timing when a user does not exist.
_DUMMY_HASH = _password_hash.hash("dummy-password-for-timing")


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        _password_hash.verify(password, _DUMMY_HASH)
        return False
    try:
        return _password_hash.verify(password, password_hash)
    except Exception:  # malformed hash
        return False


def hash_token_id(jti: str) -> str:
    return hashlib.sha256(jti.encode()).hexdigest()


@dataclass(frozen=True)
class TokenClaims:
    subject: str  # user id
    tenant_id: str
    role: str
    token_type: TokenType
    jti: str
    expires_at: datetime


def create_token(
    *,
    user_id: str,
    tenant_id: str,
    role: str,
    token_type: TokenType,
    settings: Settings | None = None,
) -> tuple[str, TokenClaims]:
    settings = settings or get_settings()
    now = datetime.now(UTC)
    if token_type == "access":
        expires = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    else:
        expires = now + timedelta(days=settings.jwt_refresh_token_expire_days)
    jti = uuid.uuid4().hex
    payload = {
        "sub": user_id,
        "tid": tenant_id,
        "role": role,
        "type": token_type,
        "jti": jti,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "iss": settings.jwt_issuer,
    }
    token = jwt.encode(payload, settings.jwt_secret_key.get_secret_value(), algorithm=settings.jwt_algorithm)
    claims = TokenClaims(user_id, tenant_id, role, token_type, jti, expires)
    return token, claims


def decode_token(token: str, expected_type: TokenType, settings: Settings | None = None) -> TokenClaims:
    settings = settings or get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],  # never accept "none" or alg switching
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub", "jti", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired", code="TOKEN_EXPIRED") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid authentication token", code="INVALID_TOKEN") from exc
    if payload.get("type") != expected_type:
        raise AuthenticationError("Invalid token type", code="INVALID_TOKEN")
    try:
        uuid.UUID(str(payload["sub"]))
        uuid.UUID(str(payload["tid"]))
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("Invalid authentication token", code="INVALID_TOKEN") from exc
    return TokenClaims(
        subject=str(payload["sub"]),
        tenant_id=str(payload["tid"]),
        role=str(payload.get("role", "member")),
        token_type=expected_type,
        jti=str(payload["jti"]),
        expires_at=datetime.fromtimestamp(int(payload["exp"]), UTC),
    )
