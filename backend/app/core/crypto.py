"""Symmetric encryption for secrets stored at rest (connector credentials)."""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import Settings
from app.core.errors import AppError


class DecryptionError(AppError):
    code, status_code, message = "DECRYPTION_FAILED", 500, "Stored credentials could not be decrypted"


def _fernet(settings: Settings) -> Fernet:
    if settings.encryption_key is not None:
        key = settings.encryption_key.get_secret_value().encode()
    else:
        # Derive a stable key from the JWT secret (domain-separated) when no dedicated key is configured.
        digest = hashlib.sha256(b"agentic-graphrag/connector-credentials:" + settings.jwt_secret_key.get_secret_value().encode())
        key = base64.urlsafe_b64encode(digest.digest())
    return Fernet(key)


def encrypt(settings: Settings, plaintext: str) -> str:
    return _fernet(settings).encrypt(plaintext.encode()).decode()


def decrypt(settings: Settings, token: str) -> str:
    try:
        return _fernet(settings).decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise DecryptionError() from exc
