"""Registration, login, refresh-token rotation and logout."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import AuthenticationError, ConflictError
from app.core.logging import get_logger
from app.core.security import create_token, decode_token, hash_password, hash_token_id, verify_password
from app.db.redis import revoke_access_token
from app.models.tenant import Tenant
from app.models.user import RefreshToken, User
from app.schemas.auth import CreateUserRequest, RegisterRequest, TokenResponse
from app.services.audit import record_audit

logger = get_logger(__name__)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:180] or "tenant"
    return f"{slug}-{uuid.uuid4().hex[:8]}"


class AuthService:
    def __init__(self, db: AsyncSession, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    async def register(self, data: RegisterRequest) -> tuple[User, TokenResponse]:
        email = data.email.lower()
        if (await self.db.execute(select(User.id).where(User.email == email))).scalar_one_or_none():
            raise ConflictError("An account with this email already exists", code="EMAIL_ALREADY_REGISTERED")
        tenant = Tenant(name=data.tenant_name.strip(), slug=_slugify(data.tenant_name))
        self.db.add(tenant)
        await self.db.flush()
        # The first user of a new tenant administers it.
        user = User(tenant_id=tenant.id, email=email, full_name=data.full_name,
                    password_hash=hash_password(data.password), role="admin")
        self.db.add(user)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            await self.db.rollback()
            raise ConflictError("An account with this email already exists", code="EMAIL_ALREADY_REGISTERED") from exc
        record_audit(self.db, "auth.register", tenant_id=tenant.id, user_id=user.id, resource_type="user",
                     resource_id=str(user.id))
        tokens = await self._issue_tokens(user)
        await self.db.commit()
        return user, tokens

    async def create_user(self, admin: User | None, tenant_id: uuid.UUID, data: CreateUserRequest) -> User:
        email = data.email.lower()
        if (await self.db.execute(select(User.id).where(User.email == email))).scalar_one_or_none():
            raise ConflictError("An account with this email already exists", code="EMAIL_ALREADY_REGISTERED")
        user = User(tenant_id=tenant_id, email=email, full_name=data.full_name,
                    password_hash=hash_password(data.password), role=data.role)
        self.db.add(user)
        await self.db.flush()
        record_audit(self.db, "auth.user_created", tenant_id=tenant_id, user_id=admin.id if admin else None,
                     resource_type="user", resource_id=str(user.id))
        await self.db.commit()
        return user

    async def login(self, email: str, password: str) -> tuple[User, TokenResponse]:
        user = (await self.db.execute(select(User).where(User.email == email.lower()))).scalar_one_or_none()
        # verify_password runs a dummy hash for unknown users (no user-enumeration timing oracle).
        if not verify_password(password, user.password_hash if user else None) or user is None or not user.is_active:
            logger.info("login_failed")
            raise AuthenticationError("Invalid email or password", code="INVALID_CREDENTIALS")
        user.last_login_at = datetime.now(UTC)
        record_audit(self.db, "auth.login", tenant_id=user.tenant_id, user_id=user.id)
        tokens = await self._issue_tokens(user)
        await self.db.commit()
        return user, tokens

    async def refresh(self, refresh_token: str) -> TokenResponse:
        claims = decode_token(refresh_token, "refresh", self.settings)
        record = (
            await self.db.execute(select(RefreshToken).where(RefreshToken.jti_hash == hash_token_id(claims.jti)))
        ).scalar_one_or_none()
        if record is None:
            raise AuthenticationError("Invalid refresh token", code="INVALID_TOKEN")
        if record.revoked_at is not None:
            # Reuse of a rotated token => likely theft: revoke the whole family for this user.
            await self.db.execute(
                update(RefreshToken)
                .where(RefreshToken.user_id == record.user_id, RefreshToken.revoked_at.is_(None))
                .values(revoked_at=datetime.now(UTC))
            )
            record_audit(self.db, "auth.refresh_token_reuse", tenant_id=record.tenant_id, user_id=record.user_id)
            await self.db.commit()
            raise AuthenticationError("Refresh token has been revoked", code="TOKEN_REVOKED")
        user = await self.db.get(User, record.user_id)
        if user is None or not user.is_active or str(user.tenant_id) != claims.tenant_id:
            raise AuthenticationError("Invalid refresh token", code="INVALID_TOKEN")
        tokens = await self._issue_tokens(user)
        new_claims = decode_token(tokens.refresh_token, "refresh", self.settings)
        new_record = (
            await self.db.execute(select(RefreshToken).where(RefreshToken.jti_hash == hash_token_id(new_claims.jti)))
        ).scalar_one()
        record.revoked_at = datetime.now(UTC)
        record.replaced_by = new_record.id
        await self.db.commit()
        return tokens

    async def logout(self, user_id: uuid.UUID, tenant_id: uuid.UUID, access_jti: str, access_exp: datetime,
                     refresh_token: str | None) -> None:
        ttl = int((access_exp - datetime.now(UTC)).total_seconds())
        try:
            await revoke_access_token(access_jti, ttl)
        except Exception:
            logger.warning("access_token_revocation_failed")
        if refresh_token:
            try:
                claims = decode_token(refresh_token, "refresh", self.settings)
            except AuthenticationError:
                claims = None
            if claims is not None and claims.subject == str(user_id):
                await self.db.execute(
                    update(RefreshToken)
                    .where(RefreshToken.jti_hash == hash_token_id(claims.jti), RefreshToken.revoked_at.is_(None))
                    .values(revoked_at=datetime.now(UTC))
                )
        record_audit(self.db, "auth.logout", tenant_id=tenant_id, user_id=user_id)
        await self.db.commit()

    async def _issue_tokens(self, user: User) -> TokenResponse:
        access, _ = create_token(user_id=str(user.id), tenant_id=str(user.tenant_id), role=user.role,
                                 token_type="access", settings=self.settings)
        refresh, claims = create_token(user_id=str(user.id), tenant_id=str(user.tenant_id), role=user.role,
                                       token_type="refresh", settings=self.settings)
        self.db.add(RefreshToken(user_id=user.id, tenant_id=user.tenant_id, jti_hash=hash_token_id(claims.jti),
                                 expires_at=claims.expires_at))
        await self.db.flush()
        return TokenResponse(access_token=access, refresh_token=refresh,
                             expires_in=self.settings.jwt_access_token_expire_minutes * 60)
