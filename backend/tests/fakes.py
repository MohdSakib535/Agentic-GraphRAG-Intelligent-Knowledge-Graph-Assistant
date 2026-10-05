"""Test doubles.

``InMemoryGraph`` implements the GraphWriter + GraphReader interfaces over plain
Python structures, so the real ingestion pipeline, retrievers and LangGraph agent
can be exercised end-to-end in unit tests without Neo4j. It mirrors the tenant
scoping of the Cypher queries: every method filters on ``tenant_id``.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from typing import Any

from app.core.access import require_scope
from app.utils.text import normalize_name, term_set


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryGraph:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.chunks: dict[str, dict[str, Any]] = {}
        self.entities: dict[str, dict[str, Any]] = {}
        self.rels: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.mentions: set[tuple[str, str]] = set()

    # ================================================================ writer
    def upsert_document(self, tenant_id: str, document_id: str, filename: str, title: str | None, file_type: str) -> None:
        self.documents[document_id] = {"id": document_id, "tenant_id": tenant_id, "filename": filename, "title": title}

    def delete_document_chunks(self, tenant_id: str, document_id: str) -> None:
        for cid in [c for c, v in self.chunks.items() if v["tenant_id"] == tenant_id and v["document_id"] == document_id]:
            del self.chunks[cid]

    def write_chunks(self, tenant_id: str, document_id: str, chunks: list[dict[str, Any]]) -> int:
        for c in chunks:
            assert c["tenant_id"] == tenant_id
            self.chunks[c["id"]] = dict(c)
        return len(chunks)

    def set_chunk_embeddings(self, tenant_id: str, rows: list[dict[str, Any]]) -> int:
        for row in rows:
            if self.chunks.get(row["id"], {}).get("tenant_id") == tenant_id:
                self.chunks[row["id"]]["embedding"] = row["embedding"]
        return len(rows)

    def upsert_entities(self, tenant_id: str, entities: list[dict[str, Any]]) -> int:
        for e in entities:
            cur = self.entities.get(e["id"])
            if cur is None:
                self.entities[e["id"]] = {**e, "tenant_id": tenant_id, "document_ids": [e["document_id"]]}
            else:
                assert cur["tenant_id"] == tenant_id
                cur["aliases"] = sorted(set(cur.get("aliases", [])) | set(e.get("aliases", [])))
                if e["document_id"] not in cur["document_ids"]:
                    cur["document_ids"].append(e["document_id"])
        return len(entities)

    def write_mentions(self, tenant_id: str, mentions: list[tuple[str, str]]) -> int:
        for chunk_id, entity_id in mentions:
            if self.chunks[chunk_id]["tenant_id"] == tenant_id and self.entities[entity_id]["tenant_id"] == tenant_id:
                self.mentions.add((chunk_id, entity_id))
        return len(mentions)

    def upsert_relationships(self, tenant_id: str, document_id: str, rels: list[dict[str, Any]]) -> int:
        for r in rels:
            key = (r["source_id"], r["type"], r["target_id"])
            cur = self.rels.setdefault(key, {"tenant_id": tenant_id, "chunk_ids": [], "document_ids": [],
                                             "evidence": r["evidence"]})
            cur["chunk_ids"] = sorted(set(cur["chunk_ids"]) | set(r["chunk_ids"]))
            if document_id not in cur["document_ids"]:
                cur["document_ids"].append(document_id)
        return len(rels)

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        self.delete_document_chunks(tenant_id, document_id)
        self.documents.pop(document_id, None)
        prefix = f"chk_{uuid.UUID(document_id).hex}_"
        self.mentions = {(c, e) for c, e in self.mentions if not c.startswith(prefix)}
        for key in list(self.rels):
            r = self.rels[key]
            if r["tenant_id"] == tenant_id and document_id in r["document_ids"]:
                r["document_ids"].remove(document_id)
                r["chunk_ids"] = [c for c in r["chunk_ids"] if not c.startswith(prefix)]
                if not r["document_ids"]:
                    del self.rels[key]
        for eid in list(self.entities):
            e = self.entities[eid]
            if e["tenant_id"] == tenant_id and document_id in e["document_ids"]:
                e["document_ids"].remove(document_id)
                if not e["document_ids"]:
                    del self.entities[eid]
                    self.rels = {k: v for k, v in self.rels.items() if eid not in (k[0], k[2])}

    def find_by_keys(self, tenant_id: str, keys: list[str]) -> list[dict[str, Any]]:
        return [self._ent_row(e) for e in self.entities.values() if e["tenant_id"] == tenant_id and e["normalized_name"] in keys]

    def find_candidates(self, tenant_id: str, entity_type: str, tokens: list[str], limit: int) -> list[dict[str, Any]]:
        rows = [self._ent_row(e) for e in self.entities.values() if e["tenant_id"] == tenant_id and e["type"] == entity_type
                and any(t in e["normalized_name"] for t in tokens)]
        return rows[:limit]

    @staticmethod
    def _ent_row(e: dict[str, Any]) -> dict[str, Any]:
        return {k: e.get(k) for k in ("id", "name", "type", "normalized_name", "description")}

    # ================================================================ reader
    def _chunk_row(self, c: dict[str, Any], score: float) -> dict[str, Any]:
        return {"chunk_id": c["id"], "document_id": c["document_id"], "text": c["text"], "score": score,
                "source_filename": c.get("source_filename"), "page_number": c.get("page_number"),
                "section": c.get("section"), "chunk_index": c.get("chunk_index"), "document_title": c.get("document_title")}

    @staticmethod
    def _passes(c: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        f = filters or {}
        if f.get("document_ids") and c["document_id"] not in f["document_ids"]:
            return False
        return not (f.get("filenames") and c.get("source_filename") not in f["filenames"])

    @staticmethod
    def _denied() -> set[str]:
        return set(require_scope().denied_document_ids)  # fail closed, like GraphReader

    def _tenant_chunks(self, tenant_id: str) -> list[dict[str, Any]]:
        denied = self._denied()
        return [c for c in self.chunks.values() if c["tenant_id"] == tenant_id and c["document_id"] not in denied]

    async def count_chunks(self, tenant_id: str) -> int:
        return len(self._tenant_chunks(tenant_id))

    async def vector_search(self, tenant_id: str, embedding: list[float], top_k: int,
                            filters: dict[str, Any] | None = None, exact_scan_max: int = 20000) -> list[dict[str, Any]]:
        scored = [(_cos(c["embedding"], embedding), c) for c in self._tenant_chunks(tenant_id)
                  if c.get("embedding") and self._passes(c, filters)]
        scored.sort(key=lambda x: -x[0])
        return [self._chunk_row(c, (s + 1) / 2) for s, c in scored[:top_k]]  # Neo4j cosine score is in [0, 1]

    async def fulltext_chunks(self, tenant_id: str, terms: list[str], top_k: int,
                              filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        q = set(terms)
        scored = []
        for c in self._tenant_chunks(tenant_id):
            hits = len(q & term_set(c["text"]))
            if hits and self._passes(c, filters):
                scored.append((float(hits), c))
        scored.sort(key=lambda x: -x[0])
        return [self._chunk_row(c, s) for s, c in scored[:top_k]]

    async def chunks_by_ids(self, tenant_id: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
        visible = {c["id"] for c in self._tenant_chunks(tenant_id)}
        return [self._chunk_row(self.chunks[c], 1.0) for c in chunk_ids if c in visible]

    async def chunks_mentioning(self, tenant_id: str, entity_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        counts: dict[str, int] = defaultdict(int)
        visible = {c["id"] for c in self._tenant_chunks(tenant_id)}
        for c, e in self.mentions:
            if e in entity_ids and c in visible:
                counts[c] += 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]
        return [self._chunk_row(self.chunks[c], n / len(entity_ids)) for c, n in ranked]

    def _tenant_entities(self, tenant_id: str) -> list[dict[str, Any]]:
        denied = self._denied()
        return [e for e in self.entities.values()
                if e["tenant_id"] == tenant_id and any(d not in denied for d in e["document_ids"])]

    async def link_entities_exact(self, tenant_id: str, names: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for q in names:
            for e in self._tenant_entities(tenant_id):
                aliases = [a.lower() for a in e.get("aliases", [])]
                if e["normalized_name"] == q["key"] or e["name"].lower() == q["lower"] or q["lower"] in aliases:
                    out.append({"query": q["raw"], "id": e["id"], "name": e["name"], "type": e["type"],
                                "description": e.get("description"), "score": 1.0})
        return out

    async def link_entities_fuzzy(self, tenant_id: str, names: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
        out = []
        for q in names:
            for e in self._tenant_entities(tenant_id):
                hit = sum(1 for t in q["tokens"] if t in e["normalized_name"])
                if hit:
                    out.append({"query": q["raw"], "id": e["id"], "name": e["name"], "type": e["type"],
                                "description": e.get("description"), "score": hit / len(q["tokens"])})
        return sorted(out, key=lambda r: -r["score"])[:limit]

    async def entities_in_text(self, tenant_id: str, normalized_text: str) -> list[dict[str, Any]]:
        padded = f" {normalized_text} "
        out = []
        for e in self._tenant_entities(tenant_id):
            names = [e["normalized_name"]] + [normalize_name(a) for a in e.get("aliases", []) if len(a) > 2]
            if len(e["normalized_name"]) > 1 and any(f" {n} " in padded for n in names):
                out.append({"id": e["id"], "name": e["name"], "type": e["type"], "description": e.get("description"),
                            "score": 1.0})
        return out[:25]

    def _rel_rows(self, tenant_id: str) -> list[dict[str, Any]]:
        denied = self._denied()
        prefixes = tuple(require_scope().denied_chunk_prefixes)
        rows = []
        for (sid, rtype, tid), r in self.rels.items():
            if r["tenant_id"] != tenant_id or not any(d not in denied for d in r["document_ids"]):
                continue
            s, t = self.entities[sid], self.entities[tid]
            clean = not any(d in denied for d in r["document_ids"])
            rows.append({"source": s["name"], "source_type": s["type"], "relationship": rtype, "target": t["name"],
                         "target_type": t["type"], "evidence": r["evidence"] if clean else None,
                         "chunk_ids": [c for c in r["chunk_ids"] if not (prefixes and c.startswith(prefixes))],
                         "document_ids": [d for d in r["document_ids"] if d not in denied],
                         "source_id": sid, "target_id": tid, "hops": 1})
        return rows

    async def neighborhood(self, tenant_id: str, entity_ids: list[str], rel_types: list[str] | None, limit: int) -> list[dict[str, Any]]:
        return [r for r in self._rel_rows(tenant_id)
                if (r["source_id"] in entity_ids or r["target_id"] in entity_ids)
                and (not rel_types or r["relationship"] in rel_types)][:limit]

    async def common_neighbors(self, tenant_id: str, anchors: list[dict[str, Any]], min_anchors: int, limit: int = 20) -> list[dict[str, Any]]:
        ids = {a["id"] for a in anchors}
        reach: dict[str, set[str]] = defaultdict(set)
        for r in self._rel_rows(tenant_id):
            for a in anchors:
                if a.get("rels") and r["relationship"] not in a["rels"]:
                    continue
                for here, there in ((r["source_id"], r["target_id"]), (r["target_id"], r["source_id"])):
                    if here == a["id"] and there not in ids:
                        reach[there].add(a["id"])
        rows = [{"id": m, "name": self.entities[m]["name"], "type": self.entities[m]["type"], "anchors": len(a)}
                for m, a in reach.items() if len(a) >= min_anchors]
        return sorted(rows, key=lambda r: -r["anchors"])[:limit]

    async def paths_between(self, tenant_id: str, entity_ids: list[str], max_hops: int, limit: int = 25) -> list[dict[str, Any]]:
        rows = self._rel_rows(tenant_id)
        ids = set(entity_ids)
        return [{**r, "hops": 1} for r in rows if r["source_id"] in ids and r["target_id"] in ids][:limit]

    async def stats(self, tenant_id: str) -> dict[str, Any]:
        return {"entities": len(self._tenant_entities(tenant_id)), "relationships": len(self._rel_rows(tenant_id)),
                "chunks": len(self._tenant_chunks(tenant_id)),
                "documents": sum(1 for d in self.documents.values() if d["tenant_id"] == tenant_id),
                "entities_by_type": {}, "relationships_by_type": {}}


class FakeStructuredLLM:
    """Minimal stand-in for LLMClient returning scripted structured outputs (by task name)."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def structured(self, schema: Any, messages: Any, *, task: str) -> Any:
        self.calls.append(task)
        value = self.responses[task]
        return schema.model_validate(value) if isinstance(value, dict) else value

    async def astructured(self, schema: Any, messages: Any, *, task: str) -> Any:
        return self.structured(schema, messages, task=task)
