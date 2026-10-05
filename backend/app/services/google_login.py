"""Sign in with Google (OpenID Connect authorization-code flow with PKCE).

Flow: ``/auth/google/login`` stores ``state`` + PKCE verifier + nonce in Redis and redirects to Google;
``/auth/google/callback`` validates the state, exchanges the code, verifies the ID token signature
(Google JWKS), audience, issuer, expiry, nonce and ``email_verified``, provisions the user and redirects
the browser to the frontend with a short-lived single-use code; the frontend trades that code for JWTs
at ``/auth/google/exchange``. Tokens never appear in a URL, and nothing here is logged except outcomes.

Provisioning: an existing account (by Google subject, else by verified email) signs in; otherwise a user
whose Google Workspace domain (``hd``) is claimed by a tenant joins it as a member; otherwise a new tenant
is created with the user as admin, claiming their Workspace domain so colleagues join it automatically.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import AppError, AuthenticationError, AuthorizationError, NotFoundError, RedisUnavailable
from app.core.logging import get_logger
from app.db.redis import get_redis
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.auth import TokenResponse
from app.services.audit import record_audit
from app.services.auth_service import AuthService, _slugify

logger = get_logger(__name__)

GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}
STATE_TTL = 600
CODE_TTL = 120


class GoogleLoginError(AppError):
    code, status_code, message = "GOOGLE_LOGIN_FAILED", 401, "Google sign-in failed"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class GoogleLogin:
    def __init__(self, db: AsyncSession, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not settings.google_login_enabled:
            raise NotFoundError("Google sign-in is not configured", code="GOOGLE_LOGIN_DISABLED")
        self.db = db
        self.settings = settings
        self.transport = transport

    # ------------------------------------------------------------------ step 1
    async def authorization_url(self) -> str:
        state, verifier, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(64), secrets.token_urlsafe(32)
        await self._redis_set(f"oauth:google:state:{_digest(state)}", json.dumps({"v": verifier, "n": nonce}), STATE_TTL)
        params = {
            "client_id": self.settings.google_client_id, "redirect_uri": self.settings.google_redirect_uri,
            "response_type": "code", "scope": "openid email profile", "state": state, "nonce": nonce,
            "code_challenge": _b64url(hashlib.sha256(verifier.encode()).digest()), "code_challenge_method": "S256",
            "prompt": "select_account", "access_type": "online",
        }
        return f"{self.settings.google_auth_url}?{urlencode(params)}"

    # ------------------------------------------------------------------ step 2
    async def callback(self, code: str, state: str) -> str:
        """Returns a one-time code for the frontend."""
        stored = await self._redis_pop(f"oauth:google:state:{_digest(state)}")
        if not stored:
            raise GoogleLoginError("The sign-in request expired or was already used", code="GOOGLE_STATE_INVALID")
        pkce = json.loads(stored)
        claims = await self._verified_claims(await self._exchange(code, pkce["v"]), pkce["n"])
        user = await self._provision(claims)
        one_time = secrets.token_urlsafe(32)
        await self._redis_set(f"oauth:google:code:{_digest(one_time)}", str(user.id), CODE_TTL)
        return one_time

    # ------------------------------------------------------------------ step 3
    async def exchange(self, one_time_code: str) -> tuple[User, TokenResponse]:
        user_id = await self._redis_pop(f"oauth:google:code:{_digest(one_time_code)}")
        user = await self.db.get(User, user_id) if user_id else None
        if user is None or not user.is_active:
            raise AuthenticationError("Invalid or expired sign-in code", code="INVALID_SSO_CODE")
        user.last_login_at = datetime.now(UTC)
        record_audit(self.db, "auth.login", tenant_id=user.tenant_id, user_id=user.id, details={"provider": "google"})
        tokens = await AuthService(self.db, self.settings)._issue_tokens(user)
        await self.db.commit()
        return user, tokens

    # --------------------------------------------------------------- internals
    async def _exchange(self, code: str, verifier: str) -> str:
        secret = self.settings.google_client_secret.get_secret_value() if self.settings.google_client_secret else ""
        form = {"code": code, "client_id": self.settings.google_client_id, "client_secret": secret,
                "redirect_uri": self.settings.google_redirect_uri, "grant_type": "authorization_code",
                "code_verifier": verifier}
        try:
            async with httpx.AsyncClient(timeout=15, transport=self.transport) as client:
                response = await client.post(self.settings.google_token_url, data=form)
        except httpx.HTTPError as exc:
            raise GoogleLoginError("Could not reach Google", code="GOOGLE_UNREACHABLE") from exc
        if response.status_code != 200 or "id_token" not in response.json():
            logger.info("google_code_exchange_failed", extra={"status": response.status_code})
            raise GoogleLoginError("Google rejected the sign-in code", code="GOOGLE_CODE_REJECTED")
        return response.json()["id_token"]

    async def _verified_claims(self, id_token: str, nonce: str) -> dict[str, Any]:
        try:
            kid = jwt.get_unverified_header(id_token).get("kid")
            async with httpx.AsyncClient(timeout=10, transport=self.transport) as client:
                jwks = (await client.get(self.settings.google_jwks_url)).json()
            jwk = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
            if jwk is None:
                raise GoogleLoginError("Unknown Google signing key", code="GOOGLE_TOKEN_INVALID")
            claims = jwt.decode(id_token, jwt.PyJWK(jwk).key, algorithms=["RS256"], audience=self.settings.google_client_id,
                                options={"require": ["exp", "iat", "iss", "sub", "aud"]}, leeway=30)
        except (jwt.PyJWTError, httpx.HTTPError, ValueError) as exc:
            raise GoogleLoginError("The Google ID token is invalid", code="GOOGLE_TOKEN_INVALID") from exc
        if claims.get("iss") not in GOOGLE_ISSUERS or not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise GoogleLoginError("The Google ID token is invalid", code="GOOGLE_TOKEN_INVALID")
        if not claims.get("email") or claims.get("email_verified") is not True:
            raise GoogleLoginError("Your Google email address is not verified", code="GOOGLE_EMAIL_UNVERIFIED")
        domain = claims["email"].rsplit("@", 1)[-1].lower()
        allowed = [d.lower() for d in self.settings.google_allowed_domains]
        if allowed and domain not in allowed and str(claims.get("hd", "")).lower() not in allowed:
            raise AuthorizationError("Your Google account's domain is not allowed", code="GOOGLE_DOMAIN_NOT_ALLOWED")
        return claims

    async def _provision(self, claims: dict[str, Any]) -> User:
        sub, email = str(claims["sub"]), claims["email"].lower()
        hd = str(claims.get("hd") or "").lower() or None  # set only for Google Workspace accounts
        name = (claims.get("name") or "").strip()[:200] or None
        user = (await self.db.execute(select(User).where(User.google_sub == sub))).scalar_one_or_none()
        if user is None:
            user = (await self.db.execute(select(User).where(User.email == email))).scalar_one_or_none()
            if user is not None:  # Google verified this email: link the existing account
                user.google_sub = sub
                record_audit(self.db, "auth.google_linked", tenant_id=user.tenant_id, user_id=user.id)
        if user is None:
            tenant = None
            if hd:
                tenant = (await self.db.execute(select(Tenant).where(Tenant.sso_domain == hd))).scalar_one_or_none()
            role = "member"
            if tenant is None:
                tenant_name = hd or f"{name or email.split('@')[0]}'s workspace"
                tenant = Tenant(name=tenant_name, slug=_slugify(tenant_name), sso_domain=hd)
                self.db.add(tenant)
                await self.db.flush()
                role = "admin"
            user = User(tenant_id=tenant.id, email=email, full_name=name, password_hash=None,
                        auth_provider="google", google_sub=sub, role=role)
            self.db.add(user)
            try:
                await self.db.flush()
            except IntegrityError as exc:  # concurrent first sign-in
                await self.db.rollback()
                raise GoogleLoginError("Please try signing in again", code="GOOGLE_LOGIN_RACE") from exc
            record_audit(self.db, "auth.register", tenant_id=tenant.id, user_id=user.id,
                         details={"provider": "google", "role": role})
        if not user.is_active:
            raise AuthenticationError("This account is disabled", code="ACCOUNT_DISABLED")
        await self.db.commit()
        return user

    @staticmethod
    async def _redis_set(key: str, value: str, ttl: int) -> None:
        try:
            await get_redis().set(key, value, ex=ttl)
        except RedisError as exc:
            raise RedisUnavailable() from exc

    @staticmethod
    async def _redis_pop(key: str) -> str | None:
        try:
            value = await get_redis().getdel(key)  # single use
        except RedisError as exc:
            raise RedisUnavailable() from exc
        return value.decode() if isinstance(value, bytes) else value
