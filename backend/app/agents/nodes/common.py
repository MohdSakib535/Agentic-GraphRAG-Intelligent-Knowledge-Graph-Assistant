"""Helpers shared by agent nodes: streaming events, trace steps, evidence and citations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.repository import GraphReader
from app.llm.client import LLMClient
from app.retrieval.query_parsing import is_definition_question
from app.utils.text import STOPWORDS, content_terms, split_sentences, term_set, truncate

logger = get_logger(__name__)

REL_VERBS = {
    "MANAGES": "manages",
    "WORKS_ON": "works on",
    "WORKS_FOR": "works for",
    "USES": "uses",
    "BELONGS_TO": "belongs to",
    "REPORTS_TO": "reports to",
    "DEPENDS_ON": "depends on",
    "RELATED_TO": "is related to",
}
# Words that carry no evidential content in answers/questions.
GENERIC_TERMS = {
    "answer", "knowledge", "graph", "based", "base", "supporting", "fact", "show", "following", "relevant",
    "passage", "information", "according", "document", "source", "uploaded", "who", "which", "what",
}


@dataclass
class AgentDeps:
    settings: Settings
    analyzer: Any  # QueryAnalyzer
    tools: Any  # AgentTools
    reader: GraphReader
    llm: LLMClient | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def emit(event: str, data: dict[str, Any]) -> None:
    """Send a custom stream event (SSE). No-op when the graph is not being streamed."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return
    try:
        writer({"event": event, "data": data})
    except Exception:
        logger.debug("stream_writer_failed")


def step(state: dict[str, Any], name: str, started: float, **detail: Any) -> list[dict[str, Any]]:
    entry = {"step": name, "status": detail.pop("status", "done"), "detail": detail,
             "latency_ms": int((time.perf_counter() - started) * 1000)}
    return [*(state.get("trace") or []), entry]


def verbalize(fact: dict[str, Any]) -> str:
    verb = REL_VERBS.get(fact["relationship"], fact["relationship"].lower().replace("_", " "))
    return f"{fact['source']} {verb} {fact['target']}"


def key_terms(text: str) -> set[str]:
    return {t for t in term_set(text) if t not in GENERIC_TERMS}


# ------------------------------------------------------------------ evidence
@dataclass
class Evidence:
    sources: list[dict[str, Any]]  # citation entries (index starts at 1)
    chunks: list[dict[str, Any]]
    facts: list[dict[str, Any]]  # each with "citation" index
    prompt_text: str

    def text_for(self, indices: set[int] | None = None) -> str:
        parts = []
        for src in self.sources:
            if indices is None or src["index"] in indices:
                parts.append(src.get("full_text") or src.get("snippet") or "")
        for fact in self.facts:
            if indices is None or fact.get("citation") in indices:
                parts.append(verbalize(fact) + ". " + (fact.get("evidence") or ""))
        return "\n".join(parts)


async def build_evidence(state: dict[str, Any], reader: GraphReader, max_chunks: int = 8, max_facts: int = 25) -> Evidence:
    chunks = list(state.get("vector_results") or [])[:max_chunks]
    facts = [dict(f) for f in (state.get("graph_results") or [])[:max_facts]]
    sources: list[dict[str, Any]] = []
    by_chunk: dict[str, int] = {}

    def add_chunk_source(chunk: dict[str, Any]) -> int:
        cid = chunk["chunk_id"]
        if cid in by_chunk:
            return by_chunk[cid]
        index = len(sources) + 1
        by_chunk[cid] = index
        sources.append({
            "index": index,
            "kind": "chunk",
            "chunk_id": cid,
            "document_id": chunk.get("document_id"),
            "source_filename": chunk.get("source_filename"),
            "page_number": chunk.get("page_number"),
            "section": chunk.get("section"),
            "snippet": truncate(chunk.get("text") or "", 300),
            "full_text": chunk.get("text") or "",
        })
        return index

    for chunk in chunks:
        add_chunk_source(chunk)
    # Graph facts cite the chunk that states them; fetch any evidence chunk not already retrieved.
    missing = [c for f in facts for c in (f.get("chunk_ids") or [])[:1] if c not in by_chunk]
    if missing:
        try:
            rows = await reader.chunks_by_ids(state["tenant_id"], list(dict.fromkeys(missing)))
        except Exception:
            rows = []
        extra = {r["chunk_id"]: r for r in rows}
    else:
        extra = {}
    for fact in facts:
        cid = next((c for c in (fact.get("chunk_ids") or []) if c in by_chunk or c in extra), None)
        if cid is not None:
            fact["citation"] = by_chunk.get(cid) or add_chunk_source(extra[cid])
        else:
            index = len(sources) + 1
            sources.append({"index": index, "kind": "graph", "chunk_id": None,
                            "document_id": (fact.get("document_ids") or [None])[0], "source_filename": None,
                            "page_number": None, "section": None, "snippet": fact.get("evidence") or verbalize(fact),
                            "full_text": fact.get("evidence") or verbalize(fact)})
            fact["citation"] = index

    lines = []
    for src in sources:
        loc = ", ".join(x for x in (
            src.get("source_filename"),
            f"page {src['page_number']}" if src.get("page_number") else None,
            src.get("section"),
        ) if x)
        lines.append(f"[{src['index']}] ({loc or 'knowledge graph'}) {truncate(src['full_text'], 1500)}")
    if facts:
        lines.append("\nKnowledge-graph facts:")
        lines.extend(f"- {verbalize(f)} [{f['citation']}]" for f in facts)
    cypher_rows = state.get("cypher_rows") or []
    if cypher_rows:
        lines.append("\nGraph query results:")
        lines.extend(f"- {truncate(str(r), 300)}" for r in cypher_rows[:20])
    return Evidence(sources=sources, chunks=chunks, facts=facts, prompt_text="\n".join(lines))


def public_sources(evidence: Evidence, cited: set[int] | None = None) -> list[dict[str, Any]]:
    out = []
    for src in evidence.sources:
        if cited is not None and src["index"] not in cited:
            continue
        out.append({k: v for k, v in src.items() if k != "full_text"})
    return out


_CITATION_RE = re.compile(r"\[(\d+)\]")


def cited_indices(answer: str) -> set[int]:
    return {int(m) for m in _CITATION_RE.findall(answer)}


# ------------------------------------------------------- heuristic generation
def relevant_facts(state: dict[str, Any], facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors = {e["name"] for e in (state.get("linked_entities") or [])}
    candidates = set(state.get("answer_candidates") or [])
    hints = set(state.get("relations") or [])
    focus = anchors | candidates
    out = []
    for fact in facts:
        touches = fact["source"] in focus or fact["target"] in focus
        if touches and (not hints or fact["relationship"] in hints):
            out.append(fact)
    return out


def defined_entities(question: str, entity_names: list[str]) -> list[str]:
    """Entities the question asks to define ("What is Neo4j and ...", "Describe Project Alpha")."""
    out = []
    for name in entity_names:
        if re.search(rf"\b(?:what\s+(?:is|are)|define|describe|explain|tell\s+me\s+about)\s+(?:the\s+)?{re.escape(name)}\b",
                     question, re.I):
            out.append(name)
    return out


def extractive_sentences(question: str, evidence: Evidence, entity_names: list[str], limit: int = 3) -> list[tuple[str, int]]:
    q_terms = key_terms(question) | {t for n in entity_names for t in term_set(n)}
    if not q_terms:
        return []
    to_define = defined_entities(question, entity_names)
    scored: list[tuple[float, str, int]] = []
    for rank, src in enumerate(evidence.sources):
        if src["kind"] != "chunk":
            continue
        for sentence in split_sentences(src["full_text"]):
            terms = term_set(sentence)
            overlap = len(q_terms & terms) / len(q_terms)
            if overlap <= 0 or len(sentence) < 20:
                continue
            bonus = 0.15 if re.search(r"\b(?:is|are)\s+(?:a|an|the)\b", sentence) else 0.0
            scored.append((overlap + bonus - 0.03 * rank, sentence, src["index"]))
    scored.sort(key=lambda x: -x[0])
    chosen: list[tuple[str, int]] = []
    # Guarantee one defining sentence per entity the question asks to define.
    for name in to_define:
        pattern = re.compile(rf"\s*(?:the\s+)?{re.escape(name)}\s+(?:is|are)\s+(?:a|an|the)\b", re.I)
        best = next(((s, i) for _, s, i in scored if pattern.match(s)), None)
        if best and best not in chosen and len(chosen) < limit:
            chosen.append(best)
    for score, sentence, idx in scored:
        if score < 0.34 or len(chosen) >= limit:
            break
        if all(sentence != s for s, _ in chosen):
            chosen.append((sentence, idx))
    return chosen


def heuristic_answer(state: dict[str, Any], evidence: Evidence) -> str:
    question = state.get("rewritten_query") or state.get("standalone_question") or state["question"]
    strategy = state.get("retrieval_strategy", "HYBRID")
    facts = relevant_facts(state, evidence.facts)
    candidates = state.get("answer_candidates") or []
    anchors = {e["name"] for e in (state.get("linked_entities") or [])}
    bridges = {b["name"] for b in (state.get("bridges") or [])}
    lines: list[str] = []
    names = [e["name"] for e in (state.get("linked_entities") or [])]
    explanatory = strategy == "HYBRID" and is_definition_question(question)
    if explanatory:
        # "How is Redis used in Project Alpha?": lead with the explanatory passage(s).
        for sentence, idx in extractive_sentences(question, evidence, names, limit=2):
            lines.append(f"{sentence} [{idx}]")
        between = [f for f in facts if f["source"] in anchors and f["target"] in anchors]
        if between and not candidates:
            facts = between
        if lines and facts:
            lines.append("")
    if facts and strategy in {"GRAPH", "HYBRID"}:
        if candidates:
            cand = set(candidates)
            hub = (bridges - cand) or anchors
            # Facts that justify each candidate: candidate <-> hub, plus the hub <-> anchor chain.
            cand_facts = [f for f in facts if (f["source"] in cand and f["target"] in hub | anchors)
                          or (f["target"] in cand and f["source"] in hub | anchors)]
            chain = [f for f in facts if f not in cand_facts and {f["source"], f["target"]} <= (hub | anchors)
                     and {f["source"], f["target"]} & hub]
            cites = sorted({f["citation"] for f in cand_facts})
            lines.append(f"{', '.join(candidates[:10])}. " + "".join(f"[{c}]" for c in cites[:4]))
            lines.append("")
            lines.append("Supporting facts:")
            selected = cand_facts + chain
        else:
            selected = facts
        seen: set[str] = set()
        for fact in selected[:12]:
            sentence = verbalize(fact)
            if sentence not in seen:
                seen.add(sentence)
                lines.append(f"- {sentence}. [{fact['citation']}]")
    if strategy == "VECTOR" or not lines:
        for sentence, idx in extractive_sentences(question, evidence, names):
            lines.append(f"{sentence} [{idx}]")
    elif strategy == "HYBRID" and not candidates and not explanatory:
        extra = extractive_sentences(question, evidence, names, limit=1)
        if extra:
            lines.append("")
            lines.append(f"Related passage: {extra[0][0]} [{extra[0][1]}]")
    return "\n".join(lines).strip()


# ------------------------------------------------------ heuristic verification
# Capitalised names; internal dots allowed ("Node.js") but never a trailing sentence period.
_NAME_RE = re.compile(r"\b[A-Z][A-Za-z0-9+#\-]*(?:\.[A-Za-z0-9]+)*(?:\s[A-Z][A-Za-z0-9+#\-]*(?:\.[A-Za-z0-9]+)*)*")
_SKIP_LINE = re.compile(r"^(?:supporting facts|related passage|sources?)\s*:?\s*$", re.I)


def claims_of(answer: str) -> list[str]:
    claims = []
    for line in answer.split("\n"):
        line = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        line = re.sub(r"^(?:related passage|supporting facts)\s*:\s*", "", line, flags=re.I)
        if not line or _SKIP_LINE.match(line):
            continue
        for sentence in split_sentences(line):
            cleaned = _CITATION_RE.sub("", sentence).strip()
            if len([t for t in content_terms(cleaned) if t not in STOPWORDS]) >= 1:
                claims.append(cleaned)
    return claims


def heuristic_verify(question: str, answer: str, evidence: Evidence, threshold: float) -> dict[str, Any]:
    cited = cited_indices(answer)
    valid_cited = {c for c in cited if any(s["index"] == c for s in evidence.sources)}
    support_text = evidence.text_for(valid_cited or None)
    support_terms = term_set(support_text)
    support_lower = support_text.lower()
    claims = claims_of(answer)
    unsupported = []
    for claim in claims:
        terms = key_terms(claim)
        names = _NAME_RE.findall(claim)
        term_ok = not terms or len(terms & support_terms) / len(terms) >= 0.7
        names_ok = all(n.lower() in support_lower for n in names if n.lower() not in STOPWORDS)
        if not (term_ok and names_ok):
            unsupported.append(claim)
    support = 1.0 - (len(unsupported) / len(claims)) if claims else 0.0
    q_terms = key_terms(question)
    answer_terms = term_set(answer) | term_set(support_text)
    relevant = not q_terms or bool(q_terms & answer_terms)
    has_sources = bool(valid_cited) and len(valid_cited) == len(cited)
    passed = bool(claims) and has_sources and relevant and support >= threshold
    return {
        "passed": passed,
        "support_score": round(support, 3),
        "has_citations": has_sources,
        "relevant": relevant,
        "unsupported_claims": unsupported[:5],
        "claims_checked": len(claims),
        "method": "heuristic",
    }
