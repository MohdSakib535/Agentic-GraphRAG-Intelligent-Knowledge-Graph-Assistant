"""Typed client for the FastAPI backend.

All business logic lives in the backend; the Streamlit UI only calls these methods.
Tokens are passed in the Authorization header and refreshed transparently once
when the access token has expired.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


@dataclass
class Tokens:
    access_token: str
    refresh_token: str


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id

    def __str__(self) -> str:
        rid = f" (request {self.request_id})" if self.request_id else ""
        return f"{self.message}{rid}"


class APIClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        tokens: Tokens | None = None,
        on_tokens_refreshed: Callable[[Tokens], None] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tokens = tokens
        self.on_tokens_refreshed = on_tokens_refreshed
        self.timeout = timeout

    # ----------------------------------------------------------------- core
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.tokens.access_token}"} if self.tokens else {}

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        try:
            body = response.json()
            err = body.get("error", {})
            raise APIError(response.status_code, err.get("code", "HTTP_ERROR"), err.get("message", response.text),
                           body.get("request_id"))
        except (ValueError, AttributeError):
            raise APIError(response.status_code, "HTTP_ERROR", response.text[:300]) from None

    def _request(self, method: str, path: str, *, retry: bool = True, **kwargs: Any) -> Any:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, f"{self.base_url}{path}", headers=self._headers(), **kwargs)
        except httpx.HTTPError as exc:
            raise APIError(503, "BACKEND_UNREACHABLE", f"Cannot reach the API: {type(exc).__name__}") from exc
        if response.status_code == 401 and retry and self.tokens and self._try_refresh():
            return self._request(method, path, retry=False, **kwargs)
        if response.status_code >= 400:
            self._raise(response)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def _try_refresh(self) -> bool:
        if not self.tokens:
            return False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/auth/refresh", json={"refresh_token": self.tokens.refresh_token})
        except httpx.HTTPError:
            return False
        if response.status_code != 200:
            return False
        data = response.json()
        self.tokens = Tokens(data["access_token"], data["refresh_token"])
        if self.on_tokens_refreshed:
            self.on_tokens_refreshed(self.tokens)
        return True

    # ----------------------------------------------------------------- auth
    def register(self, email: str, password: str, tenant_name: str, full_name: str | None = None) -> dict[str, Any]:
        data = self._request("POST", "/auth/register", retry=False, json={
            "email": email, "password": password, "tenant_name": tenant_name, "full_name": full_name or None})
        self.tokens = Tokens(data["access_token"], data["refresh_token"])
        return data

    def login(self, email: str, password: str) -> dict[str, Any]:
        data = self._request("POST", "/auth/login", retry=False, json={"email": email, "password": password})
        self.tokens = Tokens(data["access_token"], data["refresh_token"])
        return data

    def logout(self) -> None:
        if self.tokens:
            try:
                self._request("POST", "/auth/logout", retry=False, json={"refresh_token": self.tokens.refresh_token})
            finally:
                self.tokens = None

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/auth/me")

    # ------------------------------------------------------------ documents
    def upload_document(self, filename: str, content: bytes, content_type: str | None) -> dict[str, Any]:
        files = {"file": (filename, content, content_type or "application/octet-stream")}
        return self._request("POST", "/documents/upload", files=files)

    def list_documents(self, limit: int = 100, offset: int = 0, status: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._request("GET", "/documents", params=params)

    def document(self, document_id: str) -> dict[str, Any]:
        return self._request("GET", f"/documents/{document_id}")

    def document_status(self, document_id: str) -> dict[str, Any]:
        return self._request("GET", f"/documents/{document_id}/status")

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    def reprocess_document(self, document_id: str) -> dict[str, Any]:
        return self._request("POST", f"/documents/{document_id}/reprocess")

    def document_stats(self) -> dict[str, Any]:
        return self._request("GET", "/documents/stats")

    # ----------------------------------------------------------------- chat
    def chat(self, message: str, conversation_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"message": message}
        if conversation_id:
            body["conversation_id"] = conversation_id
        return self._request("POST", "/chat", json=body)

    def chat_stream(self, message: str, conversation_id: str | None = None, _retry: bool = True) -> Iterator[dict[str, Any]]:
        """Yield ``{"event": name, "data": dict}`` items parsed from the SSE stream."""
        body: dict[str, Any] = {"message": message}
        if conversation_id:
            body["conversation_id"] = conversation_id
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, read=None)) as client:
                with client.stream("POST", f"{self.base_url}/chat/stream", json=body, headers=self._headers()) as response:
                    if response.status_code == 401 and _retry and self._try_refresh():
                        yield from self.chat_stream(message, conversation_id, _retry=False)
                        return
                    if response.status_code >= 400:
                        response.read()
                        self._raise(response)
                    event, data_lines = "message", []
                    for line in response.iter_lines():
                        if line.startswith(":"):
                            continue  # keep-alive comment
                        if not line:
                            if data_lines:
                                raw = "\n".join(data_lines)
                                try:
                                    payload = json.loads(raw)
                                except ValueError:
                                    payload = {"raw": raw}
                                yield {"event": event, "data": payload}
                            event, data_lines = "message", []
                        elif line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
        except httpx.HTTPError as exc:
            raise APIError(503, "BACKEND_UNREACHABLE", f"Streaming connection failed: {type(exc).__name__}") from exc

    def conversations(self) -> list[dict[str, Any]]:
        return self._request("GET", "/chat/conversations")

    def conversation_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/chat/conversations/{conversation_id}/messages")

    def delete_conversation(self, conversation_id: str) -> None:
        self._request("DELETE", f"/chat/conversations/{conversation_id}")

    # ------------------------------------------------------- search / graph
    def search(self, query: str, strategy: str = "HYBRID", top_k: int = 8) -> dict[str, Any]:
        return self._request("POST", "/search", json={"query": query, "strategy": strategy, "top_k": top_k})

    def graph_stats(self) -> dict[str, Any]:
        return self._request("GET", "/graph/stats")

    def graph_entities(self, q: str | None = None, types: list[str] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if q:
            params["q"] = q
        if types:
            params["types"] = types
        return self._request("GET", "/graph/entities", params=params)

    def graph_entity(self, entity_id: str) -> dict[str, Any]:
        return self._request("GET", f"/graph/entities/{entity_id}")

    def graph_subgraph(self, entity_id: str | None = None, types: list[str] | None = None, limit: int = 300,
                       depth: int = 1) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "depth": depth}
        if entity_id:
            params["entity_id"] = entity_id
        if types:
            params["types"] = types
        return self._request("GET", "/graph/subgraph", params=params)

    # ------------------------------------------------------------ evaluation
    def run_evaluation(self, systems: list[str] | None = None, categories: list[str] | None = None,
                       limit: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if systems:
            body["systems"] = systems
        if categories:
            body["categories"] = categories
        if limit:
            body["limit"] = limit
        return self._request("POST", "/evaluation/run", json=body)

    def evaluation_results(self, run_id: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/evaluation/results", params={"run_id": run_id} if run_id else None)

    def evaluation_dataset(self) -> dict[str, Any]:
        return self._request("GET", "/evaluation/dataset")

    # --------------------------------------------------------------- system
    def settings(self) -> dict[str, Any]:
        return self._request("GET", "/settings")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health/ready", retry=False)
