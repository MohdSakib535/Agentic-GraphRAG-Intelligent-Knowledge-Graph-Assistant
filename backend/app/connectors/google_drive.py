"""Google Drive client authenticated with a service account (OAuth 2.0 JWT bearer grant).

Share a Drive folder with the service account's e-mail address, then create a connector with
the folder id. Only the read-only Drive scope is requested.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

from app.core.config import Settings
from app.core.errors import AppError, ValidationFailed

SCOPE = "https://www.googleapis.com/auth/drive.readonly"
FOLDER = "application/vnd.google-apps.folder"
GOOGLE_DOC = "application/vnd.google-apps.document"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
# Drive MIME type -> (our file type, export MIME type or None for a direct download)
SUPPORTED: dict[str, tuple[str, str | None]] = {
    "application/pdf": ("pdf", None),
    DOCX: ("docx", None),
    "text/plain": ("txt", None),
    "text/markdown": ("md", None),
    "text/x-markdown": ("md", None),
    GOOGLE_DOC: ("docx", DOCX),
}


class DriveError(AppError):
    code, status_code, message = "GOOGLE_DRIVE_ERROR", 502, "Google Drive request failed"


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str
    mime_type: str
    modified_time: str
    size: int | None
    path: str

    @property
    def file_type(self) -> str:
        return SUPPORTED[self.mime_type][0]

    @property
    def filename(self) -> str:
        ext = f".{self.file_type}"
        return self.name if self.name.lower().endswith(ext) else f"{self.name}{ext}"


def parse_service_account(raw: str) -> dict[str, Any]:
    try:
        info = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationFailed("Service-account credentials must be the JSON key file", code="INVALID_CREDENTIALS") from exc
    missing = [k for k in ("type", "client_email", "private_key") if not info.get(k)]
    if missing or info.get("type") != "service_account":
        raise ValidationFailed(f"Not a service-account key (missing: {', '.join(missing) or 'type'})",
                               code="INVALID_CREDENTIALS")
    return info


class GoogleDriveClient:
    def __init__(self, settings: Settings, service_account: dict[str, Any], transport: httpx.BaseTransport | None = None,
                 max_bytes: int | None = None) -> None:
        self.settings = settings
        self.sa = service_account
        self.api = settings.google_drive_api_url.rstrip("/")
        self.token_url = service_account.get("token_uri") or settings.google_token_url
        self.max_bytes = max_bytes or settings.max_upload_bytes
        self._http = httpx.Client(timeout=60, transport=transport)
        self._token: tuple[str, float] | None = None

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ auth
    def _access_token(self) -> str:
        if self._token and self._token[1] - 60 > time.time():
            return self._token[0]
        now = int(time.time())
        assertion = jwt.encode(
            {"iss": self.sa["client_email"], "scope": SCOPE, "aud": self.token_url, "iat": now, "exp": now + 3600},
            self.sa["private_key"], algorithm="RS256",
            headers={"kid": self.sa["private_key_id"]} if self.sa.get("private_key_id") else None,
        )
        response = self._http.post(self.token_url, data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion})
        if response.status_code != 200:
            raise DriveError(f"Google token exchange failed ({response.status_code})", code="GOOGLE_AUTH_FAILED")
        body = response.json()
        self._token = (body["access_token"], time.time() + int(body.get("expires_in", 3600)))
        return self._token[0]

    def _get(self, url: str, **params: Any) -> httpx.Response:
        response = self._http.get(url, params=params, headers={"Authorization": f"Bearer {self._access_token()}"})
        if response.status_code == 404:
            raise DriveError("Drive folder or file not found - is it shared with the service account?",
                             code="GOOGLE_DRIVE_NOT_FOUND")
        if response.status_code in (401, 403):
            raise DriveError("Access denied by Google Drive - share the folder with the service account",
                             code="GOOGLE_DRIVE_FORBIDDEN")
        if response.status_code >= 400:
            raise DriveError(f"Google Drive API error ({response.status_code})")
        return response

    # ----------------------------------------------------------------- files
    def _children(self, folder_id: str) -> Iterator[dict[str, Any]]:
        token = None
        while True:
            params = {"q": f"'{folder_id}' in parents and trashed = false", "pageSize": 100,
                      "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                      "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}
            if token:
                params["pageToken"] = token
            body = self._get(f"{self.api}/files", **params).json()
            yield from body.get("files", [])
            token = body.get("nextPageToken")
            if not token:
                return

    def list_files(self, folder_id: str, max_files: int, max_depth: int = 5) -> tuple[list[DriveFile], list[str]]:
        """Supported files under the folder (recursively) and names of skipped ones."""
        files: list[DriveFile] = []
        skipped: list[str] = []
        stack = [(folder_id, "", 0)]
        while stack and len(files) < max_files:
            current, prefix, depth = stack.pop()
            for item in self._children(current):
                path = f"{prefix}/{item['name']}" if prefix else item["name"]
                if item["mimeType"] == FOLDER:
                    if depth < max_depth:
                        stack.append((item["id"], path, depth + 1))
                    continue
                if item["mimeType"] not in SUPPORTED:
                    skipped.append(path)
                    continue
                size = int(item["size"]) if item.get("size") else None
                if size is not None and size > self.max_bytes:
                    skipped.append(f"{path} (too large)")
                    continue
                files.append(DriveFile(item["id"], item["name"], item["mimeType"], item["modifiedTime"], size, path))
                if len(files) >= max_files:
                    break
        return files, skipped

    def download(self, file: DriveFile) -> bytes:
        export = SUPPORTED[file.mime_type][1]
        if export:
            response = self._get(f"{self.api}/files/{file.id}/export", mimeType=export)
        else:
            response = self._get(f"{self.api}/files/{file.id}", alt="media", supportsAllDrives="true")
        if len(response.content) > self.max_bytes:
            raise DriveError(f"{file.name} exceeds the upload size limit", code="FILE_TOO_LARGE")
        return response.content

    def check_access(self, folder_id: str) -> dict[str, Any]:
        meta = self._get(f"{self.api}/files/{folder_id}", fields="id, name, mimeType", supportsAllDrives="true").json()
        if meta.get("mimeType") != FOLDER:
            raise ValidationFailed("The id does not refer to a Drive folder", code="NOT_A_FOLDER")
        return meta
