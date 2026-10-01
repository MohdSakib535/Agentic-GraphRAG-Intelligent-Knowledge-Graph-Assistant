from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.schemas.common import ORMModel

_PASSWORD_RULES = (
    (re.compile(r"[a-z]"), "a lowercase letter"),
    (re.compile(r"[A-Z]"), "an uppercase letter"),
    (re.compile(r"\d"), "a digit"),
)


def _check_password(value: str) -> str:
    missing = [desc for pattern, desc in _PASSWORD_RULES if not pattern.search(value)]
    if missing:
        raise ValueError("Password must contain " + ", ".join(missing))
    return value


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=200)
    tenant_name: str = Field(min_length=2, max_length=200, description="Organisation / workspace name")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "rahul@techcorp.com",
                "password": "Str0ngPassw0rd",
                "full_name": "Rahul Sharma",
                "tenant_name": "TechCorp",
            }
        }
    )

    _pw = field_validator("password")(_check_password)


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=200)
    role: str = Field(default="member", pattern="^(member|admin)$")

    _pw = field_validator("password")(_check_password)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "rahul@techcorp.com", "password": "Str0ngPassw0rd"}}
    )


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=4096)


class LogoutRequest(BaseModel):
    refresh_token: str | None = Field(default=None, max_length=4096)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(ORMModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    full_name: str | None
    role: str
    is_active: bool
    created_at: datetime


class TenantOut(ORMModel):
    id: uuid.UUID
    name: str
    slug: str


class MeResponse(BaseModel):
    user: UserOut
    tenant: TenantOut


class AuthResponse(TokenResponse):
    user: UserOut
