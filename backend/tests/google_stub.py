"""Fake Google OAuth token endpoint + Drive v3 API for connector and Google-login tests."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from typing import Any

import jwt
import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, Response

FOLDER = "application/vnd.google-apps.folder"


def new_rsa_key() -> tuple[str, Any]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    return pem, key.public_key()


class FakeGoogle:
    def __init__(self) -> None:
        self.sa_private_key, self.sa_public_key = new_rsa_key()
        self.files: dict[str, dict[str, Any]] = {}  # id -> {name, mimeType, parent, content, modifiedTime}
        self.token_requests = 0
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.app = self._build()
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def service_account_json(self) -> str:
        return json.dumps({"type": "service_account", "client_email": "rag-sync@test.iam.gserviceaccount.com",
                           "private_key": self.sa_private_key, "private_key_id": "k1",
                           "token_uri": f"{self.base}/token"})

    def add(self, file_id: str, name: str, mime: str, parent: str, content: bytes = b"", modified: str | None = None) -> None:
        self.files[file_id] = {"id": file_id, "name": name, "mimeType": mime, "parent": parent, "content": content,
                               "modifiedTime": modified or time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())}

    def _build(self) -> FastAPI:
        app = FastAPI()

        def authorised(request: Request) -> bool:
            return request.headers.get("authorization") == "Bearer fake-drive-token"

        @app.post("/token")
        def token(grant_type: str = Form(...), assertion: str = Form(...)) -> JSONResponse:
            self.token_requests += 1
            claims = jwt.decode(assertion, self.sa_public_key, algorithms=["RS256"], audience=f"{self.base}/token")
            assert grant_type == "urn:ietf:params:oauth:grant-type:jwt-bearer"
            assert claims["scope"] == "https://www.googleapis.com/auth/drive.readonly"
            return JSONResponse({"access_token": "fake-drive-token", "expires_in": 3600})

        @app.get("/drive/v3/files")
        def list_files(request: Request, q: str) -> JSONResponse:
            if not authorised(request):
                return JSONResponse({"error": "unauthorised"}, status_code=401)
            parent = re.match(r"'([^']+)' in parents", q).group(1)  # type: ignore[union-attr]
            children = [{k: v for k, v in f.items() if k in {"id", "name", "mimeType", "modifiedTime"}}
                        | ({"size": str(len(f["content"]))} if f["mimeType"] != FOLDER else {})
                        for f in self.files.values() if f["parent"] == parent]
            return JSONResponse({"files": children})

        @app.get("/drive/v3/files/{file_id}")
        def get_file(request: Request, file_id: str, alt: str | None = None) -> Response:
            if not authorised(request):
                return JSONResponse({"error": "unauthorised"}, status_code=401)
            f = self.files.get(file_id)
            if f is None:
                return JSONResponse({"error": "notFound"}, status_code=404)
            if alt == "media":
                return Response(f["content"])
            return JSONResponse({"id": f["id"], "name": f["name"], "mimeType": f["mimeType"]})

        @app.get("/drive/v3/files/{file_id}/export")
        def export(request: Request, file_id: str, mimeType: str) -> Response:  # noqa: N803
            if not authorised(request):
                return JSONResponse({"error": "unauthorised"}, status_code=401)
            return Response(self.files[file_id]["content"])

        return app

    def __enter__(self) -> FakeGoogle:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
