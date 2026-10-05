"""Fake Google OAuth token endpoint + Drive v3 API for connector and Google-login tests."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import socket
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import jwt
import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
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
        self.oidc_private_key, self.oidc_public_key = new_rsa_key()
        self.codes: dict[str, dict[str, Any]] = {}  # authorization code -> {claims, challenge, verifier_ok}
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

    def authorize(self, auth_url: str, *, sub: str, email: str, hd: str | None = None, name: str = "Test User",
                  email_verified: bool = True, nonce: str | None = None, client_id: str | None = None) -> tuple[str, str]:
        """Simulates the user consenting at Google. Returns (authorization code, state)."""
        params = {k: v[0] for k, v in parse_qs(urlparse(auth_url).query).items()}
        assert params["code_challenge_method"] == "S256" and params["response_type"] == "code"
        now = int(time.time())
        claims = {"iss": "https://accounts.google.com", "aud": client_id or params["client_id"], "sub": sub,
                  "email": email, "email_verified": email_verified, "name": name, "iat": now, "exp": now + 600,
                  "nonce": nonce if nonce is not None else params["nonce"], **({"hd": hd} if hd else {})}
        code = f"code-{len(self.codes)}-{sub}"
        self.codes[code] = {"claims": claims, "challenge": params["code_challenge"],
                            "redirect_uri": params["redirect_uri"]}
        return code, params["state"]

    def _build(self) -> FastAPI:
        app = FastAPI()

        def authorised(request: Request) -> bool:
            return request.headers.get("authorization") == "Bearer fake-drive-token"

        @app.get("/oauth2/v3/certs")
        def jwks() -> JSONResponse:
            jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.oidc_public_key))
            return JSONResponse({"keys": [{**jwk, "kid": "oidc-1", "alg": "RS256", "use": "sig"}]})

        @app.post("/token")
        async def token(request: Request) -> JSONResponse:
            form = dict(await request.form())
            if form.get("grant_type") == "authorization_code":
                grant = self.codes.pop(form.get("code", ""), None)
                verifier = form.get("code_verifier", "")
                challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
                if grant is None or grant["challenge"] != challenge or grant["redirect_uri"] != form.get("redirect_uri") \
                        or not form.get("client_secret"):
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)
                id_token = jwt.encode(grant["claims"], self.oidc_private_key, algorithm="RS256", headers={"kid": "oidc-1"})
                return JSONResponse({"id_token": id_token, "access_token": "x", "expires_in": 3600})
            return service_account_token(form.get("grant_type", ""), form.get("assertion", ""))

        def service_account_token(grant_type: str, assertion: str) -> JSONResponse:
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
