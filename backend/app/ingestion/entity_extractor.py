"""Entity extraction.

Two interchangeable implementations share the same validated output models:

* :class:`LLMGraphExtractor` (in ``relationship_extractor``) asks an
  OpenAI-compatible model for structured JSON - every item is re-validated here.
* :class:`HeuristicEntityExtractor` is a deterministic, offline recogniser built
  from gazetteers, typed surface patterns and relation-driven typing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.logging import get_logger
from app.graph.schema import ENTITY_TYPES, EntityType
from app.utils.text import split_sentences

logger = get_logger(__name__)

MAX_NAME_LENGTH = 120


class ExtractedEntity(BaseModel):
    """Validated entity. Constructed from untrusted LLM output - be strict."""

    name: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)
    type: EntityType
    description: str = Field(default="", max_length=600)
    source_chunk: str | None = None

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        value = " ".join(value.replace("\n", " ").split()).strip(" .,;:'\"`()[]{}")
        if not value or not re.search(r"[A-Za-z0-9]", value):
            raise ValueError("entity name must contain alphanumeric characters")
        if re.search(r"[<>{}$\\]", value):
            raise ValueError("entity name contains forbidden characters")
        return value

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            for etype in ENTITY_TYPES:
                if etype.lower() == value.strip().lower():
                    return etype
        return value

    @field_validator("description")
    @classmethod
    def _clean_description(cls, value: str) -> str:
        return " ".join(value.split())


def validate_entities(raw_items: list[Any], source_chunk: str | None = None) -> list[ExtractedEntity]:
    """Validate untrusted entity dicts one by one; invalid items are dropped, not fatal."""
    valid: list[ExtractedEntity] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_items or []:
        if isinstance(item, BaseModel):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        try:
            entity = ExtractedEntity.model_validate({**item, "source_chunk": source_chunk})
        except ValidationError:
            logger.debug("dropped_invalid_entity")
            continue
        key = (entity.name.lower(), entity.type.value)
        if key not in seen:
            seen.add(key)
            valid.append(entity)
    return valid


# ============================================================ heuristic recogniser
TECHNOLOGIES = {
    "kafka": "Kafka", "apache kafka": "Kafka", "redis": "Redis", "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB", "neo4j": "Neo4j", "fastapi": "FastAPI",
    "django": "Django", "flask": "Flask", "react": "React", "angular": "Angular", "vue": "Vue",
    "kubernetes": "Kubernetes", "k8s": "Kubernetes", "docker": "Docker", "terraform": "Terraform",
    "aws": "AWS", "azure": "Azure", "gcp": "GCP", "google cloud": "GCP", "python": "Python", "java": "Java",
    "golang": "Go", "rust": "Rust", "javascript": "JavaScript", "typescript": "TypeScript", "node.js": "Node.js",
    "nodejs": "Node.js", "spark": "Spark", "apache spark": "Spark", "hadoop": "Hadoop", "airflow": "Airflow",
    "apache airflow": "Airflow", "elasticsearch": "Elasticsearch", "rabbitmq": "RabbitMQ", "celery": "Celery",
    "graphql": "GraphQL", "grpc": "gRPC", "langchain": "LangChain", "langgraph": "LangGraph",
    "pytorch": "PyTorch", "tensorflow": "TensorFlow", "snowflake": "Snowflake", "bigquery": "BigQuery",
    "jenkins": "Jenkins", "github actions": "GitHub Actions", "prometheus": "Prometheus", "grafana": "Grafana",
    "nginx": "Nginx", "kotlin": "Kotlin", "scala": "Scala", "spring boot": "Spring Boot", "s3": "S3",
    "amazon s3": "S3", "dynamodb": "DynamoDB", "cassandra": "Cassandra", "clickhouse": "ClickHouse",
    "pandas": "Pandas", "numpy": "NumPy", "streamlit": "Streamlit", "sqlalchemy": "SQLAlchemy",
    "pgvector": "pgvector", "faiss": "FAISS", "kafka streams": "Kafka Streams", "debezium": "Debezium",
    "flink": "Flink", "apache flink": "Flink", "openai": "OpenAI", "llm": "LLM", "kibana": "Kibana",
    "memcached": "Memcached", "sqlite": "SQLite", "oracle": "Oracle", "linux": "Linux", "git": "Git",
    "helm": "Helm", "argo cd": "Argo CD", "argocd": "Argo CD", "istio": "Istio", "zookeeper": "ZooKeeper",
    "kafka connect": "Kafka Connect", "minio": "MinIO", "superset": "Superset", "dbt": "dbt",
}
# Words that are also common English - require a capitalised surface form.
_AMBIGUOUS_TECH = {"react", "rust", "spark", "flask", "vue", "go", "oracle", "celery", "helm", "git", "scala"}

LOCATIONS = {
    "bangalore", "bengaluru", "mumbai", "delhi", "new delhi", "pune", "hyderabad", "chennai", "kolkata",
    "noida", "gurgaon", "gurugram", "london", "new york", "san francisco", "seattle", "berlin", "paris",
    "singapore", "dubai", "tokyo", "sydney", "toronto", "amsterdam", "dublin", "india", "usa",
    "united states", "germany", "france", "uk", "united kingdom", "canada", "australia", "japan",
}
_COMPANY_SUFFIX = r"(?:Inc|Corp|Corporation|Ltd|Limited|LLC|GmbH|Technologies|Labs|Systems|Solutions|Group)"
_ROLE_WORDS = (
    r"(?:developer|engineer|manager|architect|lead|analyst|scientist|designer|director|cto|ceo|cfo|vp|"
    r"intern|consultant|administrator|devops|sre|owner|head|tester|programmer)"
)
_NAME = r"[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2}"
_STOP_CAPS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She", "They", "We", "I", "You", "Our",
    "In", "On", "At", "For", "With", "And", "But", "Or", "If", "When", "While", "As", "By", "From", "To",
    "Of", "All", "Each", "Every", "Some", "Both", "Also", "Then", "There", "Here", "His", "Her", "Their",
    "Its", "My", "Your", "After", "Before", "During", "Since", "Project", "Team", "Department", "Overview",
    "Summary", "Introduction", "Section", "Page", "Note", "Who", "What", "Which", "Where", "Why", "How",
    "Currently", "Together", "Additionally", "However", "Today", "Yesterday", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April",
    "May", "June", "July", "August", "September", "October", "November", "December", "Senior", "Junior",
    "Lead", "Chief", "Mr", "Ms", "Mrs", "Dr", "Architecture", "Engineering", "Data", "Platform",
    "Meanwhile", "Furthermore", "Moreover", "Finally", "Recently", "Previously", "Later", "Now", "Initially",
}
_HEAD_WORDS = {"Project", "Team", "Department", "Division", "Group", "Inc", "Corp", "Corporation", "Ltd"}

_PROJECT_RE = re.compile(r"\b[Pp]roject\s+([A-Z][A-Za-z0-9\-]+(?:\s[A-Z][a-z0-9]+)?)")
_PROJECT_SUFFIX_RE = re.compile(r"\b(?:the\s+)?([A-Z][A-Za-z0-9\-]+)\s+project\b")
_COMPANY_RE = re.compile(rf"\b([A-Z][\w&]*(?:\s[A-Z][\w&]*){{0,3}}\s{_COMPANY_SUFFIX})\.?(?![\w])")
_CAMEL_COMPANY_RE = re.compile(r"\b([A-Z][a-z]+(?:Corp|Tech|Soft|Labs|Works|Systems|Data|Cloud|Bank|Pay))\b")
_DEPARTMENT_RE = re.compile(
    r"\b((?:[A-Z][a-z]+\s){0,2}[A-Z][a-z]+)\s+((?i:department|team|division))\b"
)
_PRODUCT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:\s[A-Z][A-Za-z0-9]+)?)\s+(?:product|app|application)\b")
_ROLE_BEFORE_NAME_RE = re.compile(rf"\b(?i:{_ROLE_WORDS})s?\s+({_NAME})\b")
_NAME_COMMA_ROLE_RE = re.compile(
    rf"\b({_NAME}),?\s+(?:is\s+)?(?:a|an|the|our)\s+(?:[a-z\-]+\s){{0,3}}(?i:{_ROLE_WORDS})\b"
)
_NAME_AS_ROLE_RE = re.compile(rf"\b({_NAME})\s+(?:is|was|serves as|works as|joined as)\s+(?:a|an|the)\s+")
_PERSON_SUBJECT_RE = re.compile(
    rf"\b({_NAME})\s+(?:also\s+|currently\s+)?(?:manages|managed|leads|led|heads|oversees|works|worked|"
    r"reports|reported|contributes|contributed|is working|joined|owns|develops|is responsible|is employed|"
    r"is assigned|collaborates|mentors|work|manage|lead|report|contribute|own|develop|collaborate)\b"
)
_COORDINATED_SUBJECTS_RE = re.compile(
    rf"\b({_NAME})(?:,\s*({_NAME}))?,?\s+and\s+({_NAME})\s+(?:also\s+|both\s+)?"
    r"(?:work|manage|lead|report|contribute|collaborate|are|were)\b"
)
_PERSON_OBJECT_RE = re.compile(
    rf"\b(?:reports to|reported to|managed by|led by|headed by|owned by|overseen by|developed by|"
    rf"maintained by|built by|mentored by|works with|collaborates with)\s+({_NAME})\b"
)
_LOCATION_CONTEXT_RE = re.compile(r"\b(?:based in|located in|office in|offices in|headquartered in)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)")
_DEFINITION_RE = r"\b{name}\b\s+(?:is|are|was|refers to|provides)\s+(?:a|an|the)?\s*[^.]{{5,220}}\."


@dataclass
class EntityMention:
    name: str
    type: str
    start: int
    end: int


@dataclass
class HeuristicEntityExtractor:
    """Deterministic NER tuned for enterprise documents (people, projects, technologies...)."""

    known: dict[str, str] = field(default_factory=dict)  # surface form -> entity type

    def prime(self, texts: list[str]) -> None:
        """Document-level pass: learn person/company names so later chunks type them consistently."""
        for text in texts:
            for regex in (_ROLE_BEFORE_NAME_RE, _NAME_COMMA_ROLE_RE, _PERSON_SUBJECT_RE, _PERSON_OBJECT_RE):
                for m in regex.finditer(text):
                    name = self._clean_person(m.group(1))
                    if name and self._typed_elsewhere(name) is None:
                        self.known.setdefault(name, EntityType.PERSON.value)
            for m in _COORDINATED_SUBJECTS_RE.finditer(text):
                for group in m.groups():
                    name = self._clean_person(group) if group else None
                    if name and self._typed_elsewhere(name) is None and name not in self.known:
                        self.known[name] = EntityType.PERSON.value
            for m in _NAME_AS_ROLE_RE.finditer(text):
                tail = text[m.end() : m.end() + 80].lower()
                if re.match(rf"(?:[a-z\-]+\s){{0,3}}{_ROLE_WORDS}", tail):
                    name = self._clean_person(m.group(1))
                    if name and self._typed_elsewhere(name) is None:
                        self.known.setdefault(name, EntityType.PERSON.value)
            for m in _COMPANY_RE.finditer(text):
                self.known.setdefault(m.group(1).strip(), EntityType.COMPANY.value)
            for m in _CAMEL_COMPANY_RE.finditer(text):
                if m.group(1).lower() not in TECHNOLOGIES:
                    self.known.setdefault(m.group(1), EntityType.COMPANY.value)
            for m in re.finditer(r"\b(?:works for|worked for|works at|employed by|joined)\s+([A-Z][A-Za-z0-9&]+)", text):
                cand = m.group(1)
                if cand not in self.known and cand.lower() not in TECHNOLOGIES and cand not in _STOP_CAPS:
                    self.known[cand] = EntityType.COMPANY.value
        self._consolidate_first_names()

    def _consolidate_first_names(self) -> None:
        """Drop a bare first name when it unambiguously refers to a known full name."""
        persons = [n for n, t in self.known.items() if t == EntityType.PERSON.value]
        for name in persons:
            if " " in name:
                continue
            full = [p for p in persons if " " in p and p.split()[0] == name]
            if len(full) == 1:
                del self.known[name]

    def _clean_person(self, name: str) -> str | None:
        if any(t in _HEAD_WORDS for t in name.split()):
            return None
        tokens = name.split()
        while tokens and tokens[0] in _STOP_CAPS:
            tokens = tokens[1:]
        # Drop trailing tokens that are actually other entity heads (e.g. "Amit Project").
        cleaned = " ".join(tokens).strip()
        if not cleaned or cleaned.lower() in TECHNOLOGIES or cleaned.lower() in LOCATIONS:
            return None
        if re.fullmatch(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2}", cleaned) is None:
            return None
        return cleaned

    @staticmethod
    def _typed_elsewhere(name: str) -> str | None:
        if name.lower() in TECHNOLOGIES:
            return EntityType.TECHNOLOGY.value
        if name.lower() in LOCATIONS:
            return EntityType.LOCATION.value
        return None

    def mentions(self, text: str) -> list[EntityMention]:
        found: list[EntityMention] = []

        def add(name: str, etype: str, start: int, end: int) -> None:
            found.append(EntityMention(name.strip(), etype, start, end))

        for m in _PROJECT_RE.finditer(text):
            add(f"Project {m.group(1)}", EntityType.PROJECT.value, m.start(), m.end())
        for m in _PROJECT_SUFFIX_RE.finditer(text):
            if m.group(1) not in _STOP_CAPS and m.group(1).lower() not in TECHNOLOGIES:
                add(f"Project {m.group(1)}", EntityType.PROJECT.value, m.start(1), m.end())
        for m in _COMPANY_RE.finditer(text):
            add(m.group(1), EntityType.COMPANY.value, m.start(1), m.end(1))
        for m in _DEPARTMENT_RE.finditer(text):
            head = m.group(1)
            if head.split()[0] in {"The", "Our", "A", "An"}:
                head = " ".join(head.split()[1:])
            if head and head.lower() not in TECHNOLOGIES and head not in self.known:
                add(f"{head} {m.group(2).title()}", EntityType.DEPARTMENT.value, m.start(1), m.end())
        for m in _PRODUCT_RE.finditer(text):
            if m.group(1).lower() not in TECHNOLOGIES and m.group(1) not in _STOP_CAPS:
                add(m.group(1), EntityType.PRODUCT.value, m.start(1), m.end(1))
        # Technologies (longest surface forms first so "Apache Kafka" beats "Kafka").
        for surface in sorted(TECHNOLOGIES, key=len, reverse=True):
            pattern = re.compile(rf"(?<![\w.]){re.escape(surface)}(?![\w])", re.IGNORECASE)
            for m in pattern.finditer(text):
                matched = m.group(0)
                if surface in _AMBIGUOUS_TECH and not matched[:1].isupper():
                    continue
                add(TECHNOLOGIES[surface], EntityType.TECHNOLOGY.value, m.start(), m.end())
        for loc in LOCATIONS:
            for m in re.finditer(rf"(?<![\w]){re.escape(loc)}(?![\w])", text, re.IGNORECASE):
                if m.group(0)[:1].isupper():
                    add(m.group(0).title() if not m.group(0).isupper() else m.group(0), EntityType.LOCATION.value,
                        m.start(), m.end())
        for m in _LOCATION_CONTEXT_RE.finditer(text):
            add(m.group(1), EntityType.LOCATION.value, m.start(1), m.end(1))
        # Known names (persons/companies discovered in the priming pass).
        for name, etype in self.known.items():
            for m in re.finditer(rf"(?<![\w]){re.escape(name)}(?![\w])", text):
                add(name, etype, m.start(), m.end())
            # Possessives / first-name references to a known full name.
            if etype == EntityType.PERSON.value and " " in name:
                first = name.split()[0]
                if sum(1 for n, t in self.known.items() if t == etype and n.split()[0] == first) == 1:
                    for m in re.finditer(rf"(?<![\w]){re.escape(first)}(?![\w])(?!\s[A-Z])", text):
                        add(name, etype, m.start(), m.end())
        return _resolve_overlaps(found)

    def describe(self, name: str, text: str) -> str:
        match = re.search(_DEFINITION_RE.format(name=re.escape(name)), text)
        if match:
            return match.group(0)[:300]
        for sentence in split_sentences(text):
            if name in sentence:
                return sentence[:300]
        return ""

    def extract(self, text: str, source_chunk: str | None = None) -> list[ExtractedEntity]:
        raw = [{"name": m.name, "type": m.type, "description": self.describe(m.name, text)} for m in self.mentions(text)]
        return validate_entities(raw, source_chunk)


_TYPE_PRIORITY = {
    EntityType.PROJECT.value: 0,
    EntityType.COMPANY.value: 1,
    EntityType.DEPARTMENT.value: 2,
    EntityType.PERSON.value: 3,
    EntityType.TECHNOLOGY.value: 4,
    EntityType.PRODUCT.value: 5,
    EntityType.LOCATION.value: 6,
    EntityType.CONCEPT.value: 7,
}


def _resolve_overlaps(mentions: list[EntityMention]) -> list[EntityMention]:
    """Keep the longest (then highest-priority) mention among overlapping spans."""
    ordered = sorted(mentions, key=lambda m: (-(m.end - m.start), _TYPE_PRIORITY.get(m.type, 9), m.start))
    kept: list[EntityMention] = []
    for mention in ordered:
        if all(mention.end <= k.start or mention.start >= k.end for k in kept):
            kept.append(mention)
    return sorted(kept, key=lambda m: m.start)
