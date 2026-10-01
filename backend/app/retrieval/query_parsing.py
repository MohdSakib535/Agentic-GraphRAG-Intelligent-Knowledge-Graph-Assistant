"""Lightweight, deterministic query understanding shared by retrievers and the heuristic analyzer."""

from __future__ import annotations

import re

from app.graph.schema import EntityType, RelationType

_RELATION_CUES: dict[str, tuple[str, ...]] = {
    RelationType.MANAGES: (
        r"manag\w*", r"lead(?:s|ing)?", r"led", r"head(?:s|ed)?", r"oversee\w*", r"owner", r"owns?", r"in charge",
        r"responsible for",
    ),
    RelationType.WORKS_ON: (
        r"work(?:s|ing|ed)? on", r"developers?", r"engineers?", r"contribut\w*", r"team members?", r"members?",
        r"assigned", r"staff\w*", r"people (?:on|in)", r"who is on",
    ),
    RelationType.WORKS_FOR: (r"work(?:s|ing|ed)? (?:for|at)", r"employ\w*", r"employer", r"compan(?:y|ies)"),
    RelationType.USES: (
        r"use[sd]?", r"using", r"technolog\w*", r"tech stack", r"stack", r"built (?:with|on|using)", r"rel(?:y|ies) on",
        r"powered", r"tools?", r"frameworks?", r"databases?",
    ),
    RelationType.REPORTS_TO: (r"report(?:s|ing)?(?: to)?", r"boss", r"supervis\w*", r"direct reports?"),
    RelationType.DEPENDS_ON: (r"depend\w*", r"requires?", r"prerequisites?"),
    RelationType.BELONGS_TO: (r"belong\w*", r"part of", r"located", r"based (?:in|at)", r"which department"),
    RelationType.RELATED_TO: (r"related", r"associated", r"connect\w*", r"relationship"),
}
_COMPILED = {rel: re.compile(r"\b(?:" + "|".join(cues) + r")\b", re.IGNORECASE) for rel, cues in _RELATION_CUES.items()}

_ANSWER_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:which|what)\s+(?:\w+\s+)?(?:technolog\w*|tech|tools?|frameworks?|databases?|stack|languages?)\b", re.I), EntityType.TECHNOLOGY),
    (re.compile(r"\b(?:which|what)\s+(?:\w+\s+)?projects?\b", re.I), EntityType.PROJECT),
    (re.compile(r"\b(?:which|what)\s+(?:\w+\s+)?compan(?:y|ies)\b", re.I), EntityType.COMPANY),
    (re.compile(r"\b(?:which|what)\s+(?:\w+\s+)?(?:departments?|teams?|divisions?)\b", re.I), EntityType.DEPARTMENT),
    (re.compile(r"\b(?:which|what)\s+(?:\w+\s+)?products?\b", re.I), EntityType.PRODUCT),
    (re.compile(r"\b(?:who|whom)\b|\b(?:which|what)\s+(?:\w+\s+)?(?:developers?|engineers?|people|persons?|employees?|managers?|members?|staff)\b", re.I), EntityType.PERSON),
    (re.compile(r"\bwhere\b|\b(?:which|what)\s+(?:locations?|cities|city|countr\w+|offices?)\b", re.I), EntityType.LOCATION),
]

_DEFINITION_RE = re.compile(
    r"^\s*(?:what\s+(?:is|are|does)|explain|describe|define|tell me about|how\s+(?:does|do|is|are|can)|why\s+(?:is|are|does|do)|"
    r"what's|overview of|summari[sz]e)\b",
    re.IGNORECASE,
)
_SMALLTALK_RE = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening)|bye|goodbye|who are you|what can you do|help)\W*$",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r"\b(?:in|during|before|after|since|until|by)\s+(?:(?:19|20)\d{2}|q[1-4]|january|february|march|april|may|june|july|"
    r"august|september|october|november|december|last (?:year|month|quarter|week)|this (?:year|month|quarter))\b|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_FILENAME_RE = re.compile(r"\b([\w\-]+\.(?:pdf|docx|txt|md))\b", re.IGNORECASE)


def relation_hints(question: str) -> list[str]:
    hints = [rel.value for rel, rx in _COMPILED.items() if rx.search(question)]
    # "manager" alone also implies REPORTS_TO questions ("who is Neha's manager")
    if re.search(r"\b\w+'s\s+manager\b", question, re.IGNORECASE) and RelationType.REPORTS_TO.value not in hints:
        hints.append(RelationType.REPORTS_TO.value)
    return hints


def primary_relation(question: str) -> str | None:
    """The relation governing the answer: the earliest relational cue in the question.

    "Who manages projects that use Kafka?" -> MANAGES (USES only constrains the projects).
    """
    if re.search(r"\b\w+'s\s+(?:manager|boss|supervisor)\b", question, re.IGNORECASE):
        return RelationType.REPORTS_TO.value
    best: tuple[int, str] | None = None
    for rel, rx in _COMPILED.items():
        if rel == RelationType.RELATED_TO:
            continue
        match = rx.search(question)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), rel.value)
    return best[1] if best else None


def expected_answer_type(question: str) -> str | None:
    for pattern, etype in _ANSWER_TYPE_PATTERNS:
        if pattern.search(question):
            return etype.value
    return None


def is_definition_question(question: str) -> bool:
    return bool(_DEFINITION_RE.search(question))


def is_smalltalk(question: str) -> bool:
    return bool(_SMALLTALK_RE.match(question))


def temporal_constraints(question: str) -> list[str]:
    return [m.group(0) for m in _TEMPORAL_RE.finditer(question)]


def filename_mentions(question: str) -> list[str]:
    return [m.group(1) for m in _FILENAME_RE.finditer(question)]
